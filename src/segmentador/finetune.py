"""Module for fine-tuning pretrained segmenter models in new documents."""
import typing as t
import functools
import copy

import numpy as np
import torch
import torch.nn
import tqdm


def _label_noise_tokens(
    input_ids: t.List[int],
    labels: t.List[int],
    tokens: t.List[str],
    noise_start_id: int,
    noise_end_id: int,
) -> t.Tuple[t.List[str], t.List[int], t.List[int]]:
    """Place labels to noise sequences enclosed by `noise_start_id` and `noise_end_id`."""
    input_ids_np = np.array(input_ids, dtype=int)
    noise_start_inds = np.flatnonzero(input_ids_np == noise_start_id)

    if noise_start_inds.size > 0:
        noise_end_inds = np.hstack((np.flatnonzero(input_ids_np == noise_end_id), input_ids_np.size))

        for i_start, i_end in zip(noise_start_inds, noise_end_inds):
            if i_end > i_start + 1:
                labels[i_start + 1] = 2 if labels[i_start + 1] != -100 else -100
                if i_end + 1 < input_ids_np.size:
                    labels[i_end + 1] = 3 if labels[i_end + 1] != -100 else -100

            input_ids[i_start] = -1
            if i_end < input_ids_np.size:
                input_ids[i_end] = -1

        labels = [lab for lab, i in zip(labels, input_ids) if i >= 0]
        tokens = [tok for tok, i in zip(tokens, input_ids) if i >= 0]
        input_ids = [i for i in input_ids if i >= 0]

    assert len(input_ids) == len(labels)

    return (tokens, input_ids, labels)


def text_to_ids(
    segments: t.List[t.List[str]],
    tokenizer,
    noise_start_token: str,
    noise_end_token: str,
) -> t.Tuple[t.List[t.List[int]], t.List[t.List[int]]]:
    """Convert text segments to tokenized input ids."""
    input_ids: t.List[t.List[int]] = []
    labels: t.List[t.List[int]] = []

    tokenizer = copy.deepcopy(tokenizer)
    tokenizer.add_tokens([noise_start_token, noise_end_token], special_tokens=True)

    (noise_start_id, noise_end_id) = tokenizer.encode(
        f"{noise_start_token} {noise_end_token}",
        add_special_tokens=False,
    )

    for doc_segs in segments:
        for j, seg in enumerate(doc_segs):
            if j == 0:
                seg = f"{tokenizer.cls_token} {seg}"
            if j == len(doc_segs) - 1:
                seg = f"{seg} {tokenizer.sep_token}"

            cur_tokens = tokenizer.tokenize(seg, add_special_tokens=False)
            cur_input_ids = tokenizer.convert_tokens_to_ids(cur_tokens)

            cur_labels = [-100 if tok.startswith("##") else 0 for tok in cur_tokens]
            cur_labels[0] = 1

            if j == 0:
                cur_labels[0] = -100  # NOTE: labeling [CLS] token as '-100'.
                cur_labels[1] = 1

            if j == len(doc_segs) - 1:
                cur_labels[-1] = -100  # NOTE: labeling [SEP] token as '-100'.

            (_, cur_input_ids, cur_labels) = _label_noise_tokens(
                input_ids=cur_input_ids,
                labels=cur_labels,
                tokens=cur_tokens,
                noise_start_id=noise_start_id,
                noise_end_id=noise_end_id,
            )

            if cur_input_ids:
                input_ids.append(cur_input_ids)
                labels.append(cur_labels)

    return (input_ids, labels)


def ids_to_insts(
    seg_input_ids: t.List[int],
    seg_labels: t.List[int],
    inst_length: int,
    pad_id: int,
) -> t.Tuple[torch.Tensor, torch.Tensor]:
    """Concatenate segment input ids to form instances.

    All instances will have exactly `inst_length` length.
    """
    all_input_ids: t.List[torch.Tensor] = [[]]
    all_labels: t.List[torch.Tensor] = [[]]

    for inst, labs in zip(seg_input_ids, seg_labels):
        left_size = inst_length - len(all_input_ids[-1])

        (left_inst, right_inst) = (inst[:left_size], inst[left_size:])
        (left_labs, right_labs) = (labs[:left_size], labs[left_size:])

        all_input_ids[-1].extend(left_inst)
        all_labels[-1].extend(left_labs)

        if right_inst:
            all_input_ids.append(right_inst)
            all_labels.append(right_labs)

    half_length = inst_length // 2

    for i in range(len(all_input_ids) - 1):
        left_ids, right_ids = (all_input_ids[i][half_length:], all_input_ids[i + 1][:half_length])
        left_labs, right_labs = (all_labels[i][half_length:], all_labels[i + 1][:half_length])
        all_input_ids.append(left_ids + right_ids)
        all_labels.append(left_labs + right_labs)

    for ids, labs in zip(all_input_ids, all_labels):
        if len(ids) < inst_length:
            ids.extend((inst_length - len(ids)) * [pad_id])
            labs.extend((inst_length - len(labs)) * [-100])

    all_input_ids = torch.vstack([torch.Tensor(item) for item in all_input_ids])
    all_labels = torch.vstack([torch.Tensor(item) for item in all_labels])

    all_input_ids = all_input_ids.long()
    all_labels = all_labels.long()

    return (all_input_ids, all_labels)


def finetune(
    model: torch.nn.Module,
    tokenizer,
    segments: t.List[t.List[str]],
    is_complete_input: bool,
    *,
    lr: int = 1e-4,
    max_epochs: int = 10,
    batch_size: int = 3,
    grad_acc_its: int = 1,
    device: t.Union[str, torch.device] = "cuda:0",
    inst_length: int = 1024,
    show_progress_bar: bool = True,
    focus_on_misclassifications: bool = False,
    early_stopping_accuracy_threshold: t.Optional[float] = None,
    noise_start_token: str = "[NOISE_START]",
    noise_end_token: str = "[NOISE_END]",
):
    """Fine-tune a pretraine