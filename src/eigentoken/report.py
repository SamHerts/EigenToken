"""Small terminal formatting helpers -- no dependencies, no colour codes."""

from __future__ import annotations

from typing import Iterable, Sequence


def table(headers: Sequence[str], rows: Iterable[Sequence[object]], indent: str = "") -> str:
    rows = [[("" if c is None else str(c)) for c in r] for r in rows]
    if not rows:
        return indent + "(none)"
    widths = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))

    def fmt(cells: Sequence[str]) -> str:
        out = []
        for i, c in enumerate(cells):
            out.append(c.ljust(widths[i]) if i == 0 else c.rjust(widths[i]))
        return indent + "  ".join(out).rstrip()

    lines = [fmt(headers), indent + "  ".join("-" * w for w in widths)]
    lines += [fmt(r) for r in rows]
    return "\n".join(lines)


def heading(text: str) -> str:
    return f"\n{text}\n{'=' * len(text)}"


def sub(text: str) -> str:
    return f"\n{text}\n{'-' * len(text)}"


def num(x: float, places: int = 4) -> str:
    return f"{x:+.{places}f}" if x < 0 else f"{x:.{places}f}"


def bar(value: float, lo: float = -1.0, hi: float = 1.0, width: int = 21) -> str:
    """A tiny fixed-width position marker, for axis projections."""
    span = hi - lo
    pos = 0 if span == 0 else int(round((value - lo) / span * (width - 1)))
    pos = max(0, min(width - 1, pos))
    cells = ["."] * width
    cells[width // 2] = "|"
    cells[pos] = "#"
    return "".join(cells)
