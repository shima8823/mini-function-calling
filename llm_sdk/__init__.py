from __future__ import annotations

import torch
from huggingface_hub import hf_hub_download
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    logging,
)

logging.set_verbosity_error()

class LightweightCausalLM:
    """A lightweight utility class for experimentation with small causal LMs."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-0.6B",
        *,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        trust_remote_code: bool = True,
    ) -> None:
        self._model_id = model_id
        self._device = self._select_device(device)
        self._dtype = self._select_dtype(self._device, dtype)

        self._tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        self._model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=self._dtype,
            device_map="auto" if self._device == "cuda" else None,
            trust_remote_code=trust_remote_code,
        ).to(self._device)
        self._model.eval()
        for param in self._model.parameters():
            param.requires_grad = False

    
    def _select_device(self, candidate: str | None = None) -> str:
        if candidate:
            return candidate
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _select_dtype(self, device: str, requested: torch.dtype | None) -> torch.dtype:
        if requested is not None:
            return requested
        return torch.float16 if device in {"cuda", "mps"} else torch.float32

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        tokens = torch.tensor([input_ids], device=self._device, dtype=torch.long)
        with torch.no_grad():
            outputs = self._model(input_ids=tokens)
        logits = outputs.logits[0, -1].tolist()
        return [float(val) for val in logits]

    def get_path_to_vocabulary_json(self) -> str:
        file_name = self._tokenizer.vocab_files_names.get("vocab_file", "vocab.json")
        return hf_hub_download(repo_id=self._model_id, filename=file_name)
    
    # def get_merges and get_pretokenizer_regex
