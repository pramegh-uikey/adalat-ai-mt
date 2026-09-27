"""Document-level monotonic alignment of EN/HI sentences (vecalign-style DP).

Embeds every contiguous block of 1..max_block sentences on each side, then
runs a monotonic DP over (i, j) = (sentences consumed on src, on tgt) with a
small, fixed set of moves, maximising total gain. O(n*m) states.
"""
from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .segment import leading_para_number

#: Embeds a list of sentences/blocks into L2-normalised row vectors.
EmbedFn = Callable[[list[str]], np.ndarray]

#: Allowed (src_len, tgt_len) pair moves, beyond the (1,0)/(0,1) skips.
_MOVES: tuple[tuple[int, int], ...] = ((1, 1), (1, 2), (2, 1), (1, 3), (3, 1))


def labse_embedder(
    model_name: str = "sentence-transformers/LaBSE", cache_dir: Path | None = None
) -> EmbedFn:
    """Build an `EmbedFn` backed by LaBSE, with an on-disk cache keyed by sha1(text).

    Imports `sentence_transformers` lazily so importing this module never
    requires the (heavy, network-fetching) dependency.
    """
    from sentence_transformers import SentenceTransformer  # lazy import

    model = SentenceTransformer(model_name)

    cache: dict[str, np.ndarray] = {}
    cache_path: Path | None = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{model_name.replace('/', '_')}.pkl"
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)

    def embed(sentences: list[str]) -> np.ndarray:
        keys = [hashlib.sha1(s.encode("utf-8")).hexdigest() for s in sentences]
        missing_pos = [i for i, k in enumerate(keys) if k not in cache]
        if missing_pos:
            missing_texts = [sentences[i] for i in missing_pos]
            vecs = model.encode(
                missing_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False
            )
            for i, v in zip(missing_pos, vecs):
                cache[keys[i]] = np.asarray(v, dtype=np.float32)
            if cache_path is not None:
                with open(cache_path, "wb") as f:
                    pickle.dump(cache, f)
        return np.stack([cache[k] for k in keys])

    return embed


@dataclass
class AlignConfig:
    """Hyperparameters for `align_document`."""

    max_block: int = 3
    skip_threshold: float = 0.40
    merge_penalty: float = 0.05
    anchor_bonus: float = 0.05
    min_score: float = 0.60
    len_ratio: tuple[float, float] = (0.5, 2.5)


@dataclass
class AlignedPair:
    """One pair move from the alignment DP (skips are not represented here)."""

    src_idx: list[int]
    tgt_idx: list[int]
    en: str
    hi: str
    score: float
    kept: bool
    drop_reason: str | None
    align_type: str


@dataclass
class DocAlignment:
    """Result of aligning one document: kept+dropped pairs plus skip counts."""

    pairs: list[AlignedPair] = field(default_factory=list)
    n_src: int = 0
    n_tgt: int = 0
    n_src_skipped: int = 0
    n_tgt_skipped: int = 0


def _block_text(sentences: list[str], start: int, length: int) -> str:
    return " ".join(sentences[start : start + length])


def align_document(
    src: list[str], tgt: list[str], embed: EmbedFn, cfg: AlignConfig
) -> DocAlignment:
    """Align `src` (English sentences) to `tgt` (Hindi sentences) via monotonic DP."""
    n, m = len(src), len(tgt)
    moves = [(a, b) for (a, b) in _MOVES if a <= cfg.max_block and b <= cfg.max_block]

    src_key_index: dict[tuple[int, int], int] = {}
    tgt_key_index: dict[tuple[int, int], int] = {}
    texts: list[str] = []
    for length in range(1, cfg.max_block + 1):
        for start in range(0, n - length + 1):
            src_key_index[(start, length)] = len(texts)
            texts.append(_block_text(src, start, length))
    for length in range(1, cfg.max_block + 1):
        for start in range(0, m - length + 1):
            tgt_key_index[(start, length)] = len(texts)
            texts.append(_block_text(tgt, start, length))

    vecs = np.asarray(embed(texts)) if texts else np.zeros((0, 1))

    def src_vec(start: int, length: int) -> np.ndarray:
        return vecs[src_key_index[(start, length)]]

    def tgt_vec(start: int, length: int) -> np.ndarray:
        return vecs[tgt_key_index[(start, length)]]

    leading_src = [leading_para_number(s) for s in src]
    leading_tgt = [leading_para_number(s) for s in tgt]

    neg_inf = float("-inf")
    dp = np.full((n + 1, m + 1), neg_inf)
    dp[0, 0] = 0.0
    # choice[(i, j)] = (move_type, a, b, cos_score)
    choice: dict[tuple[int, int], tuple[str, int, int, float]] = {}

    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            best_gain = neg_inf
            best: tuple[str, int, int, float] | None = None

            for a, b in moves:
                if i - a < 0 or j - b < 0:
                    continue
                prev = dp[i - a, j - b]
                if prev == neg_inf:
                    continue
                cos = float(np.dot(src_vec(i - a, a), tgt_vec(j - b, b)))
                anchor = 0.0
                if (
                    leading_src[i - a] is not None
                    and leading_src[i - a] == leading_tgt[j - b]
                ):
                    anchor = cfg.anchor_bonus
                gain = cos - cfg.skip_threshold - cfg.merge_penalty * (a + b - 2) + anchor
                total = prev + gain
                if total > best_gain:
                    best_gain = total
                    best = ("pair", a, b, cos)

            if i - 1 >= 0 and dp[i - 1, j] != neg_inf:
                total = dp[i - 1, j]
                if total > best_gain:
                    best_gain = total
                    best = ("skip_src", 1, 0, 0.0)

            if j - 1 >= 0 and dp[i, j - 1] != neg_inf:
                total = dp[i, j - 1]
                if total > best_gain:
                    best_gain = total
                    best = ("skip_tgt", 0, 1, 0.0)

            dp[i, j] = best_gain
            choice[(i, j)] = best  # type: ignore[assignment]

    pairs_rev: list[AlignedPair] = []
    n_src_skipped = 0
    n_tgt_skipped = 0
    i, j = n, m
    while i > 0 or j > 0:
        mtype, a, b, cos = choice[(i, j)]
        if mtype == "pair":
            si, ti = i - a, j - b
            pairs_rev.append(
                AlignedPair(
                    src_idx=list(range(si, si + a)),
                    tgt_idx=list(range(ti, ti + b)),
                    en=_block_text(src, si, a),
                    hi=_block_text(tgt, ti, b),
                    score=cos,
                    kept=False,
                    drop_reason=None,
                    align_type=f"{a}-{b}",
                )
            )
            i, j = si, ti
        elif mtype == "skip_src":
            n_src_skipped += 1
            i -= 1
        else:
            n_tgt_skipped += 1
            j -= 1

    pairs = list(reversed(pairs_rev))
    for p in pairs:
        ratio_ok = len(p.en) > 0 and cfg.len_ratio[0] <= len(p.hi) / len(p.en) <= cfg.len_ratio[1]
        if p.score < cfg.min_score:
            p.kept, p.drop_reason = False, "low_score"
        elif not ratio_ok:
            p.kept, p.drop_reason = False, "len_ratio"
        else:
            p.kept, p.drop_reason = True, None

    return DocAlignment(
        pairs=pairs, n_src=n, n_tgt=m, n_src_skipped=n_src_skipped, n_tgt_skipped=n_tgt_skipped
    )
