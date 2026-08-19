"""Tiny sparkline renderer for terminal progress feedback.

Named ``ascii_graph`` rather than ``graphlib`` on purpose: the original helper
shadowed Python 3.9+'s stdlib ``graphlib`` whenever its directory landed on
sys.path, which broke unrelated imports.
"""

from __future__ import annotations

HEIGHT_CHARS = "▁▂▃▄▅▆▇█"


def draw_graph(data: list[float], description: str, bins_count: int = 100) -> None:
    if not data:
        return

    if len(data) == bins_count:
        binned = list(data)
    elif len(data) > bins_count:
        bin_size = len(data) / bins_count
        binned = []
        for i in range(bins_count):
            start = int(i * bin_size)
            end = int((i + 1) * bin_size) if i != bins_count - 1 else len(data)
            span = end - start
            binned.append(sum(data[start:end]) / span if span else 0.0)
    else:
        repetitions = max(1, bins_count // len(data))
        binned = []
        for value in data:
            binned.extend([value] * repetitions)
        binned.extend([data[-1]] * (bins_count - len(binned)))

    lo, hi = min(binned), max(binned)
    if lo == hi:
        lo -= 1

    scale = len(HEIGHT_CHARS) - 1
    normalised = [
        max(0, min(scale, round((x - lo) / (hi - lo) * scale))) for x in binned
    ]
    graph = "".join(HEIGHT_CHARS[int(v)] for v in normalised)
    print(f"{description}:\n [{graph}]\n")
