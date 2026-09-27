"""Tokenizer specs and a uniform `TokenCounter` wrapper for each backend kind.

Three backend kinds are supported (see `configs/tokenizers.yaml`):
- ``tiktoken``: OpenAI BPE encodings (`cl100k_base`, `o200k_base`).
- ``hf``: any `transformers.AutoTokenizer.from_pretrained` checkpoint.
- ``spm_pair``: a model with separate source/target SentencePiece models
  (IndicTrans2), downloaded via `huggingface_hub.hf_hub_download`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol


@dataclass
class TokenizerSpec:
    """One `configs/tokenizers.yaml` entry."""

    name: str
    kind: Literal["tiktoken", "hf", "spm_pair"]
    ident: str
    family: str
    price: str | None = None
    model_max_len: int | None = None
    src_file: str | None = None
    tgt_file: str | None = None


class TokenCounter(Protocol):
    """Uniform interface over the three tokenizer backends."""

    unk_id: int | None

    def count(self, text: str, lang: Literal["en", "hi"]) -> int:
        """Number of tokens `text` encodes to."""
        ...

    def ids(self, text: str, lang: Literal["en", "hi"]) -> list[int]:
        """Token ids for `text`."""
        ...

    def tokenize(self, text: str, lang: Literal["en", "hi"]) -> list[str]:
        """Surface token pieces for `text` (for the illustration figure)."""
        ...


class _TiktokenCounter:
    """Wraps a `tiktoken.Encoding`. Same encoding is used for EN and HI."""

    def __init__(self, ident: str) -> None:
        import tiktoken

        self._enc = tiktoken.get_encoding(ident)
        self.unk_id: int | None = None  # tiktoken BPE has no UNK; always encodable.
        self.vocab_size: int = self._enc.n_vocab

    def ids(self, text: str, lang: Literal["en", "hi"]) -> list[int]:
        return self._enc.encode(text)

    def count(self, text: str, lang: Literal["en", "hi"]) -> int:
        return len(self.ids(text, lang))

    def tokenize(self, text: str, lang: Literal["en", "hi"]) -> list[str]:
        pieces = []
        for tid in self.ids(text, lang):
            raw = self._enc.decode_single_token_bytes(tid)
            pieces.append(raw.decode("utf-8", errors="replace"))
        return pieces


class _HFCounter:
    """Wraps a `transformers` fast/slow tokenizer via `AutoTokenizer`."""

    def __init__(self, ident: str) -> None:
        from transformers import AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(ident, trust_remote_code=False)
        self.unk_id: int | None = self._tok.unk_token_id
        self.vocab_size: int = len(self._tok)

    def ids(self, text: str, lang: Literal["en", "hi"]) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False)

    def count(self, text: str, lang: Literal["en", "hi"]) -> int:
        return len(self.ids(text, lang))

    def tokenize(self, text: str, lang: Literal["en", "hi"]) -> list[str]:
        # `convert_ids_to_tokens` returns the *internal* vocabulary string,
        # which for byte-level BPE (Llama-3, GPT-2 family) is a GPT-2
        # byte-to-unicode mojibake encoding and for SentencePiece byte
        # fallback (Llama-2/TinyLlama) is a literal placeholder like
        # `<0xE0>` — neither renders as real glyphs. Decoding each id on its
        # own gives the actual surface text (or `�` when that single
        # token's bytes are not valid UTF-8 by themselves), which is what
        # should be drawn.
        return [self._tok.decode([tid]) for tid in self.ids(text, lang)]


class _SpmPairCounter:
    """Wraps two SentencePiece models, one per source/target language."""

    def __init__(self, ident: str, src_file: str, tgt_file: str) -> None:
        from huggingface_hub import hf_hub_download
        from sentencepiece import SentencePieceProcessor

        src_path = hf_hub_download(repo_id=ident, filename=src_file)
        tgt_path = hf_hub_download(repo_id=ident, filename=tgt_file)
        self._src = SentencePieceProcessor(model_file=src_path)
        self._tgt = SentencePieceProcessor(model_file=tgt_path)
        # Both SPMs share the same unk id convention (0) unless configured otherwise.
        self.unk_id: int | None = self._src.unk_id()
        # Target (Hindi) side is what an MT model actually generates.
        self.vocab_size: int = self._tgt.vocab_size()

    def _proc(self, lang: Literal["en", "hi"]) -> Any:
        return self._src if lang == "en" else self._tgt

    def ids(self, text: str, lang: Literal["en", "hi"]) -> list[int]:
        return self._proc(lang).encode(text, out_type=int)

    def count(self, text: str, lang: Literal["en", "hi"]) -> int:
        return len(self.ids(text, lang))

    def tokenize(self, text: str, lang: Literal["en", "hi"]) -> list[str]:
        return self._proc(lang).encode(text, out_type=str)


def load_counter(spec: TokenizerSpec) -> TokenCounter:
    """Instantiate the right `TokenCounter` backend for `spec.kind`."""
    if spec.kind == "tiktoken":
        return _TiktokenCounter(spec.ident)
    if spec.kind == "hf":
        return _HFCounter(spec.ident)
    if spec.kind == "spm_pair":
        if not spec.src_file or not spec.tgt_file:
            raise ValueError(f"spm_pair spec {spec.name!r} needs src_file/tgt_file")
        return _SpmPairCounter(spec.ident, spec.src_file, spec.tgt_file)
    raise ValueError(f"unknown tokenizer kind: {spec.kind!r}")
