
"""Base classes for segmenter models."""
import typing as t
import warnings
import os

import regex
import transformers
import torch
import torch.nn
import numpy as np
import numpy.typing as npt
import tqdm.auto

from . import output_handlers
from . import input_handlers
from . import finetune


class BaseSegmenter:
    """Base class for Segmenter models."""

    NUM_CLASSES = 4

    _RE_REPR_TOKENIZER_ADJUST_01 = regex.compile(r"(?<=[\(,]\s*)(?=[a-z_]+\s*=)")
    _RE_REPR_TOKENIZER_ADJUST_02 = regex.compile(r"(?<=[{,]\s*)(?=[a-z_']+\s*:)")

    def __init__(
        self,
        uri_model: str,
        uri_tokenizer: t.Optional[str] = None,
        inference_pooling_operation: str = "asymmetric-max",
        local_files_only: bool = True,
        device: str = "cpu",
        cache_dir_model: str = "./cache/models",
        cache_dir_tokenizer: str = "./cache/tokenizers",
        uri_model_extension: str = "",
        show_download_progress_bar: bool = False,
    ):
        self.local_files_only = bool(local_files_only)

        tokenizer_is_within_model = uri_model == uri_tokenizer

        if not self.local_files_only:
            if uri_tokenizer and not tokenizer_is_within_model:
                input_handlers.download_model(
                    model_name=uri_tokenizer,
                    output_dir=cache_dir_tokenizer,
                    show_progress_bar=show_download_progress_bar,
                )

            if uri_model:
                input_handlers.download_model(
                    model_name=uri_model,
                    output_dir=cache_dir_model,
                    show_progress_bar=show_download_progress_bar,
                )

        if uri_tokenizer is not None and not tokenizer_is_within_model:
            uri_tokenizer = input_handlers.get_model_uri_if_local_file(
                model_name=uri_tokenizer,
                download_dir=cache_dir_tokenizer,
                file_extension="",
            )

        uri_model = input_handlers.get_model_uri_if_local_file(
            model_name=uri_model,
            download_dir=cache_dir_model,
            file_extension=uri_model_extension,
        )

        self.uri_model = uri_model
        self.uri_tokenizer = self.uri_model if tokenizer_is_within_model else uri_tokenizer

        self._model: t.Union[torch.nn.Module, transformers.BertForTokenClassification]
        self._tokenizer: transformers.BertTokenizerFast

        if self.uri_tokenizer:
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.uri_tokenizer,
                local_files_only=self.local_files_only,
                cache_dir=cache_dir_tokenizer,
                use_fast=True,
            )

        self._moving_window_pooler = output_handlers.AutoMovingWindowPooler(
            pooling_operation=inference_pooling_operation,
        )

        self.device = device

    def __call__(
        self, *args: t.Any, **kwargs: t.Any
    ) -> t.Union[t.List[str], t.Tuple[t.List[t.Any], ...]]:
        return self.segment_legal_text(*args, **kwargs)

    def __repr__(self) -> str:
        strs: t.List[str] = []

        strs.append(f"{self.__class__.__name__} pipeline")
        strs.append(f" o Device: {self.device}")

        strs.append(" | ")
        strs.append("(1) Tokenizer:")

        text_tokenizer = str(self._tokenizer)
        text_tokenizer = self._RE_REPR_TOKENIZER_ADJUST_01.sub("\n  ", text_tokenizer)
        text_tokenizer = self._RE_REPR_TOKENIZER_ADJUST_02.sub("\n    ", text_tokenizer)
        strs.append(" | " + text_tokenizer.replace("\n", "\n |  "))

        strs.append(" | ")
        strs.append("(2) Segmenter model:")
        strs.append(" | " + str(self._model).replace("\n", "\n |  "))

        strs.append(" | ")
        strs.append("(3) Inference pooler:")
        strs.append("   " + str(self._moving_window_pooler).replace("\n", "\n |  "))

        return "\n".join(strs)

    def eval(self) -> "BaseSegmenter":
        """Set model to evaluation mode."""
        self.model.eval()
        return self

    def train(self) -> "BaseSegmenter":
        """Set model to train mode."""
        self.model.train()
        return self

    def to(self, device: t.Union[str, torch.device]) -> "BaseSegmenter":
        """Move underlying model to `device`."""
        # pylint: disable='invalid-name'
        self.model.to(device)
        return self

    @property
    def model(self) -> t.Union[torch.nn.Module, transformers.BertForTokenClassification]:
        # pylint: disable='missing-function-docstring'
        return self._model

    @property
    def tokenizer(self) -> transformers.BertTokenizerFast:
        # pylint: disable='missing-function-docstring'
        return self._tokenizer

    @property
    def RE_JUSTIFICATIVA(self) -> regex.Pattern:
        """Regular expression used to detect 'justificativa' blocks."""
        # pylint: disable='invalid-name'
        return input_handlers.InputHandlerString.RE_JUSTIFICATIVA

    @classmethod
    def preprocess_legal_text(
        cls,
        text: str,
        return_justificativa: bool = False,
        regex_justificativa: t.Optional[t.Union[str, regex.Pattern]] = None,
    ) -> t.Union[str, t.Tuple[str, t.List[str]]]:
        """Apply minimal legal text preprocessing.

        The preprocessing steps are:
        1. Coalesce all blank spaces in text;
        2. Remove all trailing and leading blank spaces; and
        3. Pre-segment text into legal text content and `justificativa`.

        Parameters
        ----------
        text : str
            Text to be preprocessed.

        return_justificativa : bool, default=False
            If True, return a tuple in the format (content, justificativa).
            If False, return only `content`.

        regex_justificativa : str, regex.Pattern or None, default=None
            Regular expression specifying how the `justificativa` portion from legal
            documents should be detected. If None, will use the pattern predefined in
            `Segmenter.RE_JUSTIFICATIVA` class attribute.

        Returns
        -------
        preprocessed_text : str
            Content from `text` after the preprocessing steps.

        justificativa_block : t.List[str]
            Detected legal text `justificativa` blocks.
            Only returned if `return_justificativa=True`.
        """
        ret = input_handlers.InputHandlerString.preprocess_legal_text(
            text=text,
            regex_justificativa=regex_justificativa,
        )

        if return_justificativa:
            return ret

        preprocessed_text, _ = ret

        return preprocessed_text

    def _set_middle_subword_label_to_noop_(
        self, input_ids: npt.NDArray[np.int32], logits: npt.NDArray[np.float64], num_tokens: int
    ) -> npt.NDArray[np.float64]:
        """Set label to NOOP class for all subwords in the middle of a whole word."""
        noop_cls_id: int

        try:
            noop_cls_id = self._model.config.label2id.get("NO_OP", 0)  # type: ignore

        except AttributeError:
            noop_cls_id = 0

        middle_subword_new_logits = np.zeros(self.NUM_CLASSES, dtype=logits.dtype)
        middle_subword_new_logits[noop_cls_id] = 1.0

        input_ids = input_ids.ravel()

        logits_shape = logits.shape
        logits = logits.reshape(-1, self.NUM_CLASSES)

        assert input_ids.size == num_tokens, (input_ids.size, num_tokens)
        assert logits.shape[0] >= num_tokens, (logits.shape[0], num_tokens)

        subwords = self._tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=False)
        middle_subword_inds = [i for i, sw in enumerate(subwords) if sw.startswith("##")]

        logits[middle_subword_inds, :] = middle_subword_new_logits

        logits = logits.reshape(*logits_shape)
        assert logits.shape == logits_shape
        return logits

    def generate_segments_from_ids(
        self,
        input_ids: t.Union[t.Sequence[int], npt.NDArray[np.int64]],
        label_ids: t.Union[t.Sequence[int], npt.NDArray[np.int64]],
        apply_postprocessing: bool = True,
        skip_special_tokens: bool = True,
        remove_noise_subsegments: bool = True,
        maximum_noise_subsegment_length: int = 25,
    ) -> t.List[str]:
        """Generate segments from input IDs and labels.

        Parameters
        ----------
        input_ids : t.Sequence[int] or npt.NDArray[np.int64]
            IDs from tokenized text from model's tokenizer.

        label_ids : t.Sequence[int] or npt.NDArray[np.int64]
            Label ids for each token, where 'label_id=1' denotes the start of a new segment,
            `label_id=2` denotes start of noise sequence (inclusive), and `noise_id=3` denotes
            end of a noise sequence (exclusive).

        apply_postprocessing : bool, default=True
            If True, remove spurious whitespaces next to punctuation marks in the output.

        remove_noise_subsegments : bool, default=True
            If True, do not include noise subsegments in the output.

        maximum_noise_subsegment_length : int, default=25
            Maximum length (in tokens) allowed for each noise subsegments in order to be removed.
            Larger noise subsegments are kept intact. This argument is useful to prevent removing
            larger chunks of text that might actually contain useful information.

        skip_special_tokens : bool, default=True
            If True, do not include tokenizer's special tokens in output ([CLS] and [SEP]).

        Returns
        -------
        segments : t.List[str]
            List containing all segments in textual form.
        """
        input_ids = np.asarray(input_ids, dtype=int).ravel()
        label_ids = np.asarray(label_ids, dtype=int).ravel()
        label_ids = label_ids[: input_ids.size]

        sep_token_id = self._tokenizer.sep_token_id

        label2id: t.Dict[str, int] = {"SEG_START": 1, "NOISE_START": 2, "NOISE_END": 3}
        try:
            label2id.update(self._model.config.label2id)
        except AttributeError:
            pass

        segment_start_inds = np.flatnonzero(label_ids == label2id["SEG_START"])
        segment_start_inds = np.hstack((0, segment_start_inds, len(input_ids)))

        segs: t.List[str] = []

        for i, i_next in zip(segment_start_inds[:-1], segment_start_inds[1:]):
            split_ = input_ids[i:i_next]

            if remove_noise_subsegments:
                split_labels = label_ids[i:i_next]
                noise_inds_start = np.flatnonzero(split_labels == label2id["NOISE_START"])
                end_index = int(np.flatnonzero(split_ == sep_token_id).min(initial=split_.size))
                noise_inds_end = np.hstack((np.flatnonzero(split_labels == label2id["NOISE_END"]), end_index))

                for n_start, n_end in zip(noise_inds_start, noise_inds_end):
                    if n_end - n_start <= maximum_noise_subsegment_length:
                        split_[n_start:n_end] = -1

                split_ = split_[split_ >= 0]

            seg = self._tokenizer.decode(split_, skip_special_tokens=skip_special_tokens)

            if seg:
                segs.append(seg)

        if apply_postprocessing:
            output_handlers.postprocessors.remove_spurious_whitespaces_(segs)

        return segs

    def _preprocess_minibatch(
        self, minibatch: transformers.BatchEncoding
    ) -> transformers.BatchEncoding:
        """Perform necessary minibatch transformations before inference.

        Can be used by subclasses. In this base class, this method is No-op/identity operator.
        """
        # pylint: disable='no-self-use'
        return minibatch

    def _predict_minibatch(self, minibatch: transformers.BatchEncoding) -> npt.NDArray[np.float64]:
        """Predict a tokenized minibatch."""
        model_out = self._model(**minibatch)
        model_out = model_out["logits"]
        model_out = model_out.cpu().numpy()

        logits: npt.NDArray[np.float64] = model_out.astype(np.float64, copy=False)

        return logits

    def segment_legal_text(
        self,
        text: t.Union[str, t.Dict[str, t.List[int]]],
        batch_size: int = 32,
        moving_window_size: int = 512,
        window_shift_size: t.Union[float, int] = 0.25,
        return_justificativa: bool = False,
        return_labels: bool = False,
        return_logits: bool = False,
        remove_noise_subsegments: bool = False,
        maximum_noise_subsegment_length: int = 25,
        apply_postprocessing: bool = True,
        show_progress_bar: bool = False,
        regex_justificativa: t.Optional[t.Union[str, regex.Pattern]] = None,
    ) -> t.Union[t.List[str], t.Tuple[t.List[t.Any], ...]]:
        """Segment legal `text`.

        The pretrained model support texts up to 1024 subwords. Texts larger than this
        value are pre-segmented into 1024 subword blocks, and each block is feed to the
        segmenter individually.

        The block size can be configured to smaller (not larger) values using the
        `moving_window_size` from `BERTSegmenter.segment_legal_text` method during inference.

        Parameters
        ----------
        text : str or t.Dict[str, t.List[int]]
            Legal text to be segmented.

        batch_size : int, default=32
            Maximum batch size feed document blocks in parallel to model. Higher values
            leads to faster inference with higher memory cost.

        moving_window_size : int, default=512
            Moving window size, the maximum number of subwords feed in simultaneously to the
            segmenter model. Higher values leads to larger contexts for each token, at the expense
            of higher memory usage.

        window_shift_size : int or float, default=0.25
            Moving window shift size.

            - If integer, specify the shift size per step exactly, and it must be in [1, 1024]
              range.
            - If float, the shift size is calculated as `window_shift_size * moving_window_size`
              (rounded up), and it must be in the (0.0, 1.0] range.

            Overlapping logits are combined using the strategy specified by the argument
            `inference_pooling_operation` in Segmenter model initialization.

            The final prediction for each token is derived from the combined logits.

        return_justificativa : bool, default=False
            If True, return contents from the 'justificativa' block from document.

        return_labels : bool, default=False
            If True, return label list for each token.

        return_logits : bool, default=False
            If True, return logit array for each token.

        remove_noise_subsegments : bool, default=False
            If True, remove all tokens between tokens classified as `noise_start` (inclusive) and
            `noise_end` or `segment` (either exclusive), whichever occurs first.

            - Tokens classified as `noise_end` are kept. In other words, they are the first
              non-noise token past the previous noise subsegment.
            - Tokens between `noise_start` and the sentence end are also removed.
            - Tokens between the sentence end and `noise_end` are kept.
            - Only the closest `noise_start` for every `noise_end` (or the sentence end) are
              considered. In other words, redundant `noise_start` tokens are ignored.

        maximum_noise_subsegment_length : int, default=25
            Maximum length (in tokens) allowed for each noise subsegments in order to be removed.
            Larger noise subsegments are kept intact. This argument is useful to prevent removing
            larger chunks of text that might actually contain useful information.

        apply_postprocessing : bool, default=True
            If True, remove spurious whitespaces next to punctuation marks in the output.

        show_progress_bar : bool, default=False
            If True, show segmentation progress bar.

        regex_justificativa : str, regex.Pattern or None, default=None
            Regular expression specifying how the `justificativa` portion from legal
            documents should be detected. If None, will use the pattern predefined in
            `Segmenter.RE_JUSTIFICATIVA` class attribute.

        Returns
        -------
        segments : t.List[str]
            Segmented legal text.

        justificativa : t.List[str]
            Detected legal text `justificativa` blocks.
            Only returned if `return_justificativa=True`.

        labels : npt.NDArray[np.int32] of shape (N,)
            Predicted labels for each token, where `N` is the length of tokenized
            document (in subword units). The `-100` labels is a special legal, and
            ignored while computing the loss function during training.
            Only returned if `return_labels=True`.

        logits : npt.NDArray[np.float64] of shape (N, C)
            Predicted logits for each token, where `N` is the length of tokenized
            document (in subword units), and `C` is equal to the `Segmenter.NUM_CLASSES`
            attribute.
            Only returned if `return_logits=True`.
        """
        if batch_size < 1:
            raise ValueError(f"'batch_size' parameter must be >= 1 (got '{batch_size}').")

        if moving_window_size < 1:
            raise ValueError(
                f"'moving_window_size' parameter must be >= 1 (got '{moving_window_size}')."
            )

        try:
            max_moving_window_size_allowed = int(
                self._model.config.max_position_embeddings  # type: ignore
            )

            if moving_window_size > max_moving_window_size_allowed:
                warnings.warn(
                    message=(
                        "'moving_window_size' is larger than model's positional embeddings "
                        f"(moving_window_size={moving_window_size}, "
                        f"max_moving_window_size_allowed={max_moving_window_size_allowed}). "
                        "Will set 'moving_window_size' to the maximum allowed value."
                    ),
                    category=UserWarning,
                )
                moving_window_size = max_moving_window_size_allowed

        except AttributeError:
            pass

        if isinstance(window_shift_size, float):
            if not 0.0 < window_shift_size <= 1.0:
                raise ValueError("If 'window_shift_size' is a float, it must be in (0, 1] range.")

            window_shift_size = int(np.ceil(moving_window_size * window_shift_size))

        if window_shift_size < 1:
            raise ValueError(
                f"'window_shift_size' parameter must be >= 1 (got '{window_shift_size}')."
            )

        if window_shift_size > moving_window_size:
            warnings.warn(
                message=(
                    f"'window_shift_size' parameter must be <= {moving_window_size} "
                    f"(got '{window_shift_size}'). "
                    f"Will set it to {moving_window_size} automatically."
                ),
                category=UserWarning,
            )
            window_shift_size = moving_window_size

        tokens, justificativa, num_tokens = input_handlers.tokenize_input(
            text=text,
            tokenizer=self.tokenizer,
            regex_justificativa=regex_justificativa,
        )

        minibatches = input_handlers.build_minibatches(
            tokens=tokens,
            num_tokens=num_tokens,
            batch_size=batch_size,
            moving_window_size=moving_window_size,
            window_shift_size=int(window_shift_size),
            pad_id=int(self._tokenizer.pad_token_id or 0),
        )

        self.eval()
        all_logits: t.List[npt.NDArray[np.float64]] = []

        with torch.no_grad():
            for minibatch in tqdm.auto.tqdm(minibatches, disable=not show_progress_bar):
                minibatch = self._preprocess_minibatch(minibatch)
                minibatch = minibatch.to(self.device)
                model_out = self._predict_minibatch(minibatch)
                all_logits.append(model_out)

        logits = np.vstack(all_logits)
        del all_logits

        logits = self._moving_window_pooler(
            logits=logits,
            window_shift_size=window_shift_size,
        )

        tokens = transformers.BatchEncoding(
            {key: val.detach().cpu().numpy() for key, val in tokens.items()}
        )

        logits = self._set_middle_subword_label_to_noop_(
            input_ids=tokens["input_ids"],
            logits=logits,
            num_tokens=num_tokens,
        )

        label_ids = logits.argmax(axis=-1)
        label_ids = label_ids.squeeze()

        label2id: t.Optional[t.Dict[str, int]]

        if remove_noise_subsegments:
            try:
                label2id = self._model.config.label2id  # type: ignore

            except AttributeError:
                label2id = None

            label_ids, (logits, *tokens_vals) = output_handlers.remove_noise_subsegments(
                label_ids,
                logits,
                *tokens.values(),
                label2id=label2id,
                maximum_noise_subsegment_length=maximum_noise_subsegment_length,
            )

            for key, val in zip(tokens.keys(), tokens_vals):
                tokens[key] = val

        segs = self.generate_segments_from_ids(
            input_ids=tokens["input_ids"],
            label_ids=label_ids,
            apply_postprocessing=apply_postprocessing,
            remove_noise_subsegments=remove_noise_subsegments,
            maximum_noise_subsegment_length=maximum_noise_subsegment_length,
        )

        label_ids = label_ids[:num_tokens]
        logits = logits.reshape(-1, self.NUM_CLASSES)
        logits = logits[:num_tokens, :]

        assert label_ids.size == logits.shape[0]

        ret = output_handlers.pack_results(
            keys=["segments", "justificativa", "labels", "logits"],
            vals=[segs, justificativa, label_ids, logits],
            inclusion=[True, return_justificativa, return_labels, return_logits],
        )

        return ret

    def finetune(
        self,
        segments: t.List[t.List[str]],
        output_uri: t.Optional[str] = None,
        output_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
        **kwargs: t.Any,
    ) -> "BaseSegmenter":
        """Finetune segmenter model for new documents.

        Parameters
        ----------

        segments : t.List[str] or t.List[t.List[str]]
            List of segments.
            If multiple documents are provided, it should be a list of lists of strings (one separate
            list per document).

        output_uri : str or None, default=None
            If provided, will save the fine-tuned model to disk in the provided path.

        output_kwargs : t.Dict[str, t.Any] or None, default=None
            Additional arguments to pass to `transformers.BertForTokenClassification.save_pretrained(...)`.
            Only used if `output_uri` is provided.

        **kwargs : t.Any
            Additional arguments for the optimization procedure.
            The optimizer used is Adam. The available options are:

            - `lr`: int, default=1e-4
                Learning rate for Adam optimizer.

            - `max_epochs`: int, default=10
                Maximum training epochs.

            - `batch_size`: int, default=3
                Training batch size.

            - `grad_acc_its`: int, default=1
                Number of gradient accumulation steps.

            - `device`: str or torch.device, default="cuda:0"
                Training device.

            - `inst_length`: int, default=1024
                Instance length. Segments are concatenated to form instances up to the length of this
                parameter. Ulysses pretrained models only support up to 1024 tokens.

            - `show_progress_bar`: bool, default=True
                If True, show progress bar during optimization.

            - `focus_on_misclassifications`: bool, default=False
                If True, progressively increases optimization weight coefficients for instances
                with misclassifications.

            - `early_stopping_accuracy_threshold`: float, default=1.0
                Accuracy threshold value for early stopping of the optimization procedure.
                Note that accuracy during optimization and inference may differ due to distinct
                inference configuration (e.g., moving window and shift size, pooling function).

            - `noise_start_token`: str, default="[NOISE_START]"
                Token to indicate noise sequence start.

            - `noise_end_token`: str, default="[NOISE_END]"
                Token to indicate noise sequence end.
        
        Returns
        -------
        self
        """
        is_bert = isinstance(self.model, transformers.BertForTokenClassification)

        self._model = finetune.finetune(
            model=self.model,
            tokenizer=self.tokenizer,
            is_complete_input=is_bert,
            segments=segments,
            **kwargs,
        )

        self.model.to(self.device)

        if output_uri:
            output_kwargs = output_kwargs or {}
            self.tokenizer.save_pretrained(output_uri)

            if is_bert:
                self.model.save_pretrained(output_uri, **output_kwargs)
            else:
                parameters_uri = os.path.join(output_uri, "model.pt")
                torch.save(self.model.state_dict(), parameters_uri)

        return self


class LSTMSegmenterTorchModule(torch.nn.Module):
    """Bidirecional LSTM Torch model for legal document segmentation."""

    def __init__(
        self,
        lstm_hidden_layer_size: int,
        lstm_num_layers: int,
        num_embeddings: int,
        pad_id: int,
        num_classes: int,
        quantize: bool = False,
    ):
        super().__init__()

        fn_factory_emb = torch.nn.quantized.Embedding if quantize else torch.nn.Embedding
        fn_factory_lstm = torch.nn.quantized.dynamic.LSTM if quantize else torch.nn.LSTM
        fn_factory_linear = torch.nn.quantized.dynamic.Linear if quantize else torch.nn.Linear

        self.embeddings = fn_factory_emb(
            num_embeddings=num_embeddings,
            embedding_dim=768,
            padding_idx=pad_id,
        )

        self.lstm = fn_factory_lstm(
            input_size=768,
            hidden_size=lstm_hidden_layer_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.lin_out = fn_factory_linear(
            2 * lstm_hidden_layer_size,
            num_classes,
        )

    def forward(self, input_ids: torch.Tensor) -> t.Dict[str, torch.Tensor]:
        # pylint: disable='missing-function-docstring'
        out = input_ids

        out = self.embeddings(out)
        out, *_ = self.lstm(out)
        out = self.lin_out(out)

        return {"logits": out}