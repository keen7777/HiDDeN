import numpy as np
from tqdm import tqdm


def inpaint_hom_diff(
    known_image_data,
    mask,
    num_iterations=500,
    tau=0.1,
):
    """
    Homogeneous diffusion inpainting.

    Parameters
    ----------
    known_image_data:
        RGB image shaped [H, W, 3].
    mask:
        Boolean mask shaped [H, W].
        True means known/retained pixel.
    """

    inpainted = known_image_data.copy()
    h = 1.0

    for channel in range(3):
        known_values = known_image_data[:, :, channel][mask]

        if known_values.size == 0:
            raise ValueError(
                "The mask contains no known pixels."
            )

        average_value = np.mean(known_values)
        inpainted[:, :, channel][~mask] = average_value

    inpainted = inpainted.astype(np.float64)
    update_mask = ~mask

    for _ in range(num_iterations):
        padded = np.pad(
            inpainted,
            ((1, 1), (1, 1), (0, 0)),
            mode="edge",
        )

        laplacian = (
            padded[2:, 1:-1, :]
            + padded[:-2, 1:-1, :]
            + padded[1:-1, 2:, :]
            + padded[1:-1, :-2, :]
            - 4.0 * padded[1:-1, 1:-1, :]
        ) / h

        inpainted[update_mask] += (
            tau * laplacian[update_mask]
        )

    return inpainted


def probabilistic_sparsification(
    image,
    initial_mask,
    density,
    tau=0.25,
    diff_iterations=500,
    p=0.1,
    q=0.1,
    rng=None,
):
    """
    Probabilistic sparsification using homogeneous diffusion.
    """

    if rng is None:
        rng = np.random.default_rng()

    mask = initial_mask.copy()

    height, width, _ = image.shape
    target_known = int(density * height * width)

    known = set(zip(*np.where(mask)))

    with tqdm(
        total=int(mask.sum()) - target_known,
        desc="Sparsification",
        unit="px",
        leave=False,
    ) as progress:
        while int(mask.sum()) > target_known:
            candidate_pool = np.array(list(known))

            if len(candidate_pool) == 0:
                break

            sample_size = max(
                1,
                int(p * len(known)),
            )

            sample_size = min(
                sample_size,
                len(candidate_pool),
            )

            selected = rng.choice(
                len(candidate_pool),
                size=sample_size,
                replace=False,
            )

            candidates = candidate_pool[selected]

            temporary_mask = mask.copy()
            temporary_mask[
                candidates[:, 0],
                candidates[:, 1],
            ] = False

            reconstructed = inpaint_hom_diff(
                image,
                temporary_mask,
                tau=tau,
                num_iterations=diff_iterations,
            )

            errors = np.mean(
                (
                    reconstructed[
                        candidates[:, 0],
                        candidates[:, 1],
                        :,
                    ]
                    - image[
                        candidates[:, 0],
                        candidates[:, 1],
                        :,
                    ]
                )
                ** 2,
                axis=1,
            )

            keep_count = int(
                (1.0 - q) * len(candidates)
            )

            # Equivalent intention to the notebook:
            # pixels with the largest errors remain known.
            if keep_count > 0:
                keep_indices = np.argsort(
                    errors
                )[-keep_count:]

                temporary_mask[
                    candidates[keep_indices, 0],
                    candidates[keep_indices, 1],
                ] = True

            # Do not go below the requested density.
            removed_coordinates = np.argwhere(
                mask & ~temporary_mask
            )

            max_removable = (
                int(mask.sum()) - target_known
            )

            if len(removed_coordinates) > max_removable:
                temporary_mask = mask.copy()

                chosen = removed_coordinates[
                    :max_removable
                ]

                temporary_mask[
                    chosen[:, 0],
                    chosen[:, 1],
                ] = False

            removed = (
                int(mask.sum())
                - int(temporary_mask.sum())
            )

            if removed <= 0:
                # Prevent an infinite loop for very small candidate sets.
                easiest = candidates[
                    np.argmin(errors)
                ]

                if int(mask.sum()) > target_known:
                    temporary_mask[
                        easiest[0],
                        easiest[1],
                    ] = False
                    removed = 1
                else:
                    break

            mask = temporary_mask
            known = set(zip(*np.where(mask)))

            progress.update(removed)

    return mask


def nonlocal_pixel_exchange(
    image,
    mask,
    nlpe_iterations=100,
    tau=0.25,
    diff_iterations=500,
    n_candidates=10,
    rng=None,
):
    """
    Nonlocal pixel exchange using homogeneous diffusion.
    """

    if rng is None:
        rng = np.random.default_rng()

    mask = mask.copy()

    inpainted = inpaint_hom_diff(
        image,
        mask,
        tau=tau,
        num_iterations=diff_iterations,
    )

    known = set(zip(*np.where(mask)))

    mse_old = np.mean(
        (inpainted - image) ** 2
    )

    with tqdm(
        total=nlpe_iterations,
        desc="NLPE",
        unit="it",
        leave=False,
    ) as progress:
        for _ in range(nlpe_iterations):
            unknown_coordinates = np.argwhere(
                ~mask
            )

            if (
                len(unknown_coordinates) == 0
                or len(known) == 0
            ):
                break

            candidate_count = min(
                n_candidates,
                len(unknown_coordinates),
            )

            selected = rng.choice(
                len(unknown_coordinates),
                size=candidate_count,
                replace=False,
            )

            candidates = unknown_coordinates[selected]

            local_errors = np.mean(
                (
                    inpainted[
                        candidates[:, 0],
                        candidates[:, 1],
                    ]
                    - image[
                        candidates[:, 0],
                        candidates[:, 1],
                    ]
                )
                ** 2,
                axis=1,
            )

            unknown_i = tuple(
                candidates[np.argmax(local_errors)]
            )

            known_list = list(known)
            known_j = known_list[
                rng.integers(len(known_list))
            ]

            test_mask = mask.copy()
            test_mask[known_j] = False
            test_mask[unknown_i] = True

            inpainted_new = inpaint_hom_diff(
                image,
                test_mask,
                tau=tau,
                num_iterations=diff_iterations,
            )

            mse_new = np.mean(
                (inpainted_new - image) ** 2
            )

            if mse_new < mse_old:
                mask = test_mask
                inpainted = inpainted_new
                mse_old = mse_new

                known.remove(known_j)
                known.add(unknown_i)

            progress.update(1)

    return mask