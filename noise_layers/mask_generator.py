import numpy as np
import torch


# ============================================================
# MASK CONVENTION CONVERSION
# ============================================================

def retention_to_removal_mask(retention_mask):
    """
    Convert a retention mask to a removal mask.

    retention_mask:
        1 = retained / known
        0 = removed / missing

    removal_mask:
        1 = removed / missing
        0 = retained / known
    """
    return 1.0 - retention_mask


def removal_to_retention_mask(removal_mask):
    """
    Convert a removal mask to a retention mask.

    removal_mask:
        1 = removed / missing
        0 = retained / known

    retention_mask:
        1 = retained / known
        0 = removed / missing
    """
    return 1.0 - removal_mask


# ============================================================
# RANDOM RECTANGLE REMOVAL MASK
# ============================================================

class RectangleRemovalMaskGenerator:
    """
    Generate random rectangle-based binary removal masks.

    Mask convention:
        1 = removed / missing / reconstructed
        0 = retained / known

    The requested removal ratio is approached using rectangular
    regions only.

    Important:
        No random single-pixel fallback is used.

    Therefore the actual removal ratio can be slightly smaller
    than the requested target. This preserves the spatial
    structure of the inpainting mask.
    """

    def __init__(
        self,
        min_mask_size=8,
        max_aspect_ratio=3.0,
        max_rectangles=50,
        max_rectangle_ratio=0.10,
        margin=1,
        seed=42,
    ):
        self.min_mask_size = min_mask_size
        self.max_aspect_ratio = max_aspect_ratio
        self.max_rectangles = max_rectangles
        self.max_rectangle_ratio = max_rectangle_ratio
        self.margin = margin

        self.rng = np.random.RandomState(seed)

    # ========================================================
    # SAMPLE ONE RECTANGLE
    # ========================================================

    def _sample_rectangle(
        self,
        H,
        W,
    ):
        available_h = H - 2 * self.margin
        available_w = W - 2 * self.margin

        if available_h <= 0 or available_w <= 0:
            raise ValueError(
                f"Image too small for margin={self.margin}: "
                f"H={H}, W={W}"
            )

        # Random aspect ratio.
        aspect_ratio = self.rng.uniform(
            1.0 / self.max_aspect_ratio,
            self.max_aspect_ratio,
        )

        # Rectangle area.
        min_area = self.min_mask_size ** 2

        max_area = max(
            min_area,
            int(
                H
                * W
                * self.max_rectangle_ratio
            ),
        )

        area = self.rng.uniform(
            min_area,
            max_area,
        )

        # Convert area + aspect ratio into width / height.
        w = int(
            np.sqrt(
                area * aspect_ratio
            )
        )

        h = int(
            np.sqrt(
                area / aspect_ratio
            )
        )

        w = max(
            1,
            min(
                w,
                available_w,
            ),
        )

        h = max(
            1,
            min(
                h,
                available_h,
            ),
        )

        # Random position.
        #
        # randint upper bound is exclusive,
        # hence the +1.
        left = self.rng.randint(
            self.margin,
            W - self.margin - w + 1,
        )

        top = self.rng.randint(
            self.margin,
            H - self.margin - h + 1,
        )

        return (
            top,
            left,
            h,
            w,
        )

    # ========================================================
    # GENERATE ONE MASK
    # ========================================================

    def generate_one(
        self,
        H,
        W,
        removal_ratio,
    ):
        """
        Generate one rectangle-based removal mask.

        Mask convention:
            1 = removed / missing
            0 = retained / known

        The requested removal ratio is approached using rectangular
        regions only. No isolated single-pixel correction is used.
        """

        if not 0.0 <= removal_ratio <= 1.0:
            raise ValueError(
                f"removal_ratio must be in [0, 1], "
                f"got {removal_ratio}"
            )

        total_pixels = H * W

        target_pixels = int(
            round(removal_ratio * total_pixels)
        )

        removal_mask = np.zeros(
            (H, W),
            dtype=np.float32,
        )

        # Minimum size of rectangles that we allow after shrinking.
        min_final_side = 3

        # Once we are this close to the target, stop.
        #
        # For 128x128:
        # 9 pixels = 0.055% of the image.
        # This difference is negligible for the experiment.
        tolerance_pixels = min_final_side ** 2

        current_pixels = 0
        accepted_rectangles = 0
        attempts = 0

        max_attempts = self.max_rectangles * 20

        while (
            current_pixels < target_pixels
            and accepted_rectangles < self.max_rectangles
            and attempts < max_attempts
        ):
            remaining_pixels = (
                target_pixels - current_pixels
            )

            # --------------------------------------------
            # IMPORTANT:
            # Don't spend thousands of attempts trying
            # to match the final few pixels.
            # --------------------------------------------
            if remaining_pixels <= tolerance_pixels:
                break

            attempts += 1

            top, left, h, w = self._sample_rectangle(
                H,
                W,
            )

            candidate_h = h
            candidate_w = w

            accepted = False

            while (
                candidate_h >= min_final_side
                and candidate_w >= min_final_side
            ):
                region = removal_mask[
                    top:top + candidate_h,
                    left:left + candidate_w
                ]

                # Number of NEW pixels this rectangle would remove.
                #
                # Avoid copying and summing the entire image.
                rectangle_area = (
                    candidate_h * candidate_w
                )

                overlap_pixels = int(
                    region.sum()
                )

                added_pixels = (
                    rectangle_area
                    - overlap_pixels
                )

                # Entire rectangle already covered.
                if added_pixels <= 0:
                    break

                if added_pixels <= remaining_pixels:
                    removal_mask[
                        top:top + candidate_h,
                        left:left + candidate_w
                    ] = 1.0

                    current_pixels += added_pixels
                    accepted_rectangles += 1
                    accepted = True

                    break

                # Rectangle adds too many pixels.
                # Shrink the longer side first.
                if candidate_h >= candidate_w:
                    candidate_h -= 1
                else:
                    candidate_w -= 1

            if not accepted:
                continue

        return removal_mask


    # ========================================================
    # GENERATE BATCH
    # ========================================================

    def generate(
        self,
        batch_size,
        H,
        W,
        removal_ratio,
        device,
    ):
        """
        Generate a batch of removal masks.

        Returns
        -------
        torch.Tensor
            Shape:
                [B, 1, H, W]

            Values:
                0 or 1

            Convention:
                1 = removed / missing
                0 = retained / known
        """

        masks = [
            self.generate_one(
                H=H,
                W=W,
                removal_ratio=removal_ratio,
            )
            for _ in range(
                batch_size
            )
        ]

        masks = np.stack(
            masks,
            axis=0,
        )

        masks = torch.tensor(
            masks,
            dtype=torch.float32,
            device=device,
        )

        return masks.unsqueeze(1)