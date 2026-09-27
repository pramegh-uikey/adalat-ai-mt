"""Plotting functions for the tokenizer study.

`plot_fertility` uses matplotlib (Latin-only labels). `plot_hindi_example`
renders a Hindi sentence tokenised by several tokenizers as coloured boxes
using Pillow + the raqm layout engine, so combining Devanagari marks shape
correctly.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)

_PALETTE = [
    (255, 209, 102),
    (6, 214, 160),
    (17, 138, 178),
    (239, 71, 111),
    (7, 59, 76),
    (149, 125, 173),
    (244, 162, 97),
    (42, 157, 143),
]

_REPLACEMENT_COLOR = (60, 60, 60)


def _has_devanagari(text: str) -> bool:
    """True if `text` contains a Devanagari-block codepoint (U+0900-U+097F)."""
    return any("ऀ" <= ch <= "ॿ" for ch in text)


def plot_fertility(
    names: Sequence[str],
    en_fertility: Sequence[float],
    hi_fertility: Sequence[float],
    out_path: Path,
) -> None:
    """Grouped horizontal bar chart of EN vs HI fertility per tokenizer.

    Rows are sorted by `hi_fertility` ascending (best/lowest first). Labels
    are Latin-script tokenizer names only (no Devanagari needed here). The
    legend sits in the upper-right corner, where the (short) top rows leave
    room, since the bottom rows are the longest bars and would otherwise be
    covered by a lower-right legend.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = sorted(range(len(names)), key=lambda i: hi_fertility[i])
    names_sorted = [names[i] for i in order]
    en_sorted = [en_fertility[i] for i in order]
    hi_sorted = [hi_fertility[i] for i in order]

    y = range(len(names_sorted))
    height = 0.35

    fig, ax = plt.subplots(figsize=(8, max(3, 0.5 * len(names_sorted))))
    ax.barh([i + height / 2 for i in y], en_sorted, height=height, label="EN fertility", color="#118AB2")
    ax.barh([i - height / 2 for i in y], hi_sorted, height=height, label="HI fertility", color="#EF476F")
    ax.set_yticks(list(y))
    ax.set_yticklabels(names_sorted)
    ax.invert_yaxis()
    ax.set_xlabel("tokens per word (fertility)")
    ax.set_title("Tokenizer fertility: English vs Hindi")
    ax.legend(loc="upper right")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_hindi_example(
    sentence: str,
    tokenizer_pieces: dict[str, list[str]],
    font_path: Path,
    out_path: Path,
    token_counts: dict[str, int] | None = None,
) -> None:
    """Render `sentence` as coloured token-piece boxes for several tokenizers.

    One row (or, for long tokenizations, several wrapped rows) per
    (name, pieces) pair in `tokenizer_pieces`, token count shown on the
    left, pieces as alternating-coloured rounded boxes. Each piece is drawn
    with whichever font actually has its glyphs: Pillow + raqm
    (`ImageFont.Layout.RAQM`) with `font_path` (NotoSansDevanagari) for
    pieces containing Devanagari text, and matplotlib's bundled DejaVuSans
    for everything else (Latin byte-fallback pieces, the `�` placeholder,
    punctuation). Pieces that are not valid UTF-8 on their own already come
    through as `�` (see `registry.TokenCounter.tokenize`).

    `token_counts`, if given, overrides `len(pieces)` for the "(N tok)"
    label (so it can be pinned to `TokenCounter.ids()`, the definition used
    everywhere else in the study, even if `tokenize()` ever drops pieces).
    """
    from PIL import Image, ImageDraw, ImageFont, features

    if not features.check("raqm"):
        logger.warning(
            "Pillow was built without raqm support; Devanagari shaping may be incorrect."
        )

    devanagari_font = ImageFont.truetype(str(font_path), size=22, layout_engine=ImageFont.Layout.RAQM)
    try:
        import matplotlib

        latin_font_path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
        label_font = ImageFont.truetype(str(latin_font_path), size=16)
        latin_piece_font = ImageFont.truetype(str(latin_font_path), size=22)
    except Exception:  # pragma: no cover - fallback only
        label_font = ImageFont.load_default()
        latin_piece_font = label_font

    def font_for(text: str) -> "ImageFont.FreeTypeFont":
        """Devanagari font if `text` needs it, else DejaVuSans (never tofu)."""
        return devanagari_font if _has_devanagari(text) else latin_piece_font

    row_h = 56
    pad = 12
    label_w = 260
    width = 1400
    title_h = 40

    # A throwaway canvas to measure box widths before the final image (whose
    # height depends on how many lines wrapping produces) is allocated.
    measure_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))

    def box_width(text: str, font: "ImageFont.FreeTypeFont") -> int:
        bbox = measure_draw.textbbox((0, 0), text, font=font)
        return max(30, (bbox[2] - bbox[0]) + 16)

    def piece_display(piece: str, i: int) -> tuple[str, tuple[int, int, int]]:
        """(display text, box colour) for the i-th piece of one tokenizer."""
        if "�" in piece:
            return "�", _REPLACEMENT_COLOR
        # SentencePiece's "▁" leading-space meta symbol has no glyph in
        # either font; drop it, the box already visually separates pieces
        # so the boundary is not lost. If nothing is left, show a blank
        # space box (still a real token, just whitespace).
        cleaned = piece.replace("▁", "")
        return (cleaned or " "), _PALETTE[i % len(_PALETTE)]

    # First pass: wrap each tokenizer's pieces into lines that fit `width`.
    Line = list[tuple[str, tuple[int, int, int], "ImageFont.FreeTypeFont", int]]
    rows: list[tuple[str, int, list[Line]]] = []
    for name, pieces in tokenizer_pieces.items():
        n_tok = len(pieces) if token_counts is None else token_counts.get(name, len(pieces))
        lines: list[Line] = [[]]
        x = label_w
        for i, piece in enumerate(pieces):
            display, color = piece_display(piece, i)
            font = font_for(display)
            box_w = box_width(display, font)
            if x + box_w > width - pad and lines[-1]:
                lines.append([])
                x = pad
            lines[-1].append((display, color, font, box_w))
            x += box_w + 6
        rows.append((name, n_tok, lines))

    total_lines = sum(len(lines) for _, _, lines in rows)
    height = title_h + row_h * total_lines + 30

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    # Title: Latin prefix in DejaVuSans, the Hindi sentence itself in the
    # Devanagari (raqm-shaped) font, side by side on the same line.
    prefix = "Example sentence: "
    draw.text((pad, 8), prefix, font=label_font, fill="black")
    prefix_bbox = draw.textbbox((pad, 8), prefix, font=label_font)
    draw.text((prefix_bbox[2] + 4, 6), sentence, font=devanagari_font, fill="black")

    y = title_h
    for name, n_tok, lines in rows:
        label = f"{name} ({n_tok} tok)"
        draw.text((pad, y + row_h // 2 - 8), label, font=label_font, fill="black")
        for line_idx, line in enumerate(lines):
            x = label_w if line_idx == 0 else pad
            for display, color, font, box_w in line:
                box = (x, y, x + box_w, y + row_h - 16)
                draw.rounded_rectangle(box, radius=8, fill=color)
                text_fill = "white" if sum(color) < 400 else "black"
                bbox = draw.textbbox((0, 0), display, font=font)
                draw.text(
                    (x + 8, y + (row_h - 16 - (bbox[3] - bbox[1])) // 2 - bbox[1]),
                    display,
                    font=font,
                    fill=text_fill,
                )
                x += box_w + 6
            y += row_h

    caption = "� = a byte-level piece that is not valid UTF-8 on its own."
    draw.text((pad, height - 24), caption, font=label_font, fill="gray")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, dpi=(150, 150))
