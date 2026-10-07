import numpy as np

def sample_points_from_mask(mask, points_per_class, ignore_index=255, rng=None):
    """
    mask: a 2D array where each number represents a class (0=road, 1=field, 2=building...)
    Returns: sparse_target (same shape, mostly zeros) and point_mask (1=labeled, 0=unknown)
    """
    if rng is None:
        rng = np.random.default_rng()

    H, W = mask.shape
    point_mask = np.zeros((H, W), dtype=np.float32)
    sparse_target = np.zeros((H, W), dtype=np.int64)

    classes = np.unique(mask)
    classes = classes[classes != ignore_index]

    for c in classes:
        ys, xs = np.where(mask == c)
        n_pick = min(points_per_class, len(ys))
        chosen = rng.choice(len(ys), size=n_pick, replace=False)
        point_mask[ys[chosen], xs[chosen]] = 1.0
        sparse_target[ys[chosen], xs[chosen]] = c

    return sparse_target, point_mask

if __name__ == "__main__":
    mask = np.zeros((20, 20), dtype=np.int64)
    mask[:7, :] = 0
    mask[7:14, :] = 1
    mask[14:, :] = 2

    sparse, pm = sample_points_from_mask(mask, points_per_class=5)
    print("Total labeled points:", pm.sum(), "(expected 15 = 3 classes x 5)")
    print("Fraction unlabeled:", round(1 - pm.sum() / (20*20), 3))