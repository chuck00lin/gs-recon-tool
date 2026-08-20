"""Pick the sharpest frames, spread evenly across the capture.

Derived from SharkWipf/nerf_dataset_preprocessing_helper (MIT, (c) 2023 Sebastiaan Meijer):
https://github.com/SharkWipf/nerf_dataset_preprocessing_helper
See THIRD_PARTY_NOTICES.md for the full licence text.

Sharpness is the variance of the Laplacian. Naively taking the globally
sharpest N frames clumps the selection into whichever part of the orbit was
best lit, which starves SfM elsewhere; splitting the sequence into groups and
taking the best of each keeps angular coverage intact.
"""

from __future__ import annotations

from typing import Optional

import cv2
from tqdm import tqdm

from .ascii_graph import draw_graph
from .grouping import describe_selection


class ImageSelector:
    def __init__(self, images: list[str]):
        self.images = images
        self.image_fm = self._compute_sharpness_values()

    def _compute_sharpness_values(self) -> list[tuple[float, str]]:
        print("Calculating image sharpness...")
        values: list[tuple[float, str]] = []
        for path in tqdm(self.images):
            image = cv2.imread(path)
            if image is None:
                print(f"  [warn] unreadable, treating as worst: {path}")
                values.append((0.0, path))
                continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            values.append((self.variance_of_laplacian(gray), path))
        return values

    @staticmethod
    def variance_of_laplacian(image) -> float:
        return float(cv2.Laplacian(image, cv2.CV_64F).var())

    @staticmethod
    def distribute_evenly(total: int, num_of_groups: int) -> tuple[list[int], float]:
        ideal = total / num_of_groups
        error = 0.0
        distribution = [0] * num_of_groups
        for i in range(num_of_groups):
            distribution[i] = int(ideal)
            error += ideal - distribution[i]
            while error >= 1.0:
                distribution[i] += 1
                error -= 1.0
        return distribution, ideal

    def generate_deleted_images_graph(self, selected_images: list[str]) -> None:
        total = len(self.images)
        if total == 0:
            return
        bins = min(100, total)
        step = max(1, total // bins)
        selected = set(selected_images)
        percentages: list[float] = []
        for i in range(bins):
            start = i * step
            end = (i + 1) * step if i < bins - 1 else total
            group = self.images[start:end]
            if not group:
                percentages.append(0.0)
                continue
            deleted = sum(1 for img in group if img not in selected)
            percentages.append(deleted / len(group) * 100)
        draw_graph(percentages, "Distribution of to-be-deleted images")

    def generate_quality_graph(self) -> None:
        draw_graph([quality for quality, _ in self.image_fm], "Distribution of image quality")

    def filter_sharpest_images(
        self,
        target_count: int,
        group_count: Optional[int] = None,
        scalar: Optional[int] = 1,
    ) -> list[str]:
        total = len(self.images)
        target_count = max(1, min(target_count, total))
        if scalar is None:
            scalar = 1
        if group_count is None:
            group_count = max(1, target_count // (2 ** (scalar - 1)))
        group_count = max(1, min(group_count, total))

        ratio = target_count / total
        print(
            f"Requested {target_count} out of {total} images "
            f"({ratio:.1%}, 1 in {total / target_count:.1f})."
        )

        group_sizes, ideal_per_group = self.distribute_evenly(total, group_count)
        per_group, ideal_selected = self.distribute_evenly(target_count, group_count)
        print(describe_selection(total, target_count, round(ideal_per_group)))

        selected: list[str] = []
        offset = 0
        for idx, size in enumerate(group_sizes):
            end = offset + size
            group = sorted(self.image_fm[offset:end], key=lambda item: item[0], reverse=True)
            selected.extend(path for _, path in group[: per_group[idx]])
            offset = end

        self.generate_deleted_images_graph(selected)
        self.generate_quality_graph()
        return selected
