import numpy as np
import torch
from torch.utils.data import Dataset
from point_sampling import sample_points_from_mask

class PointSupervisedSegDataset(Dataset):
    """Takes (image, full_mask) pairs, and at training time, secretly
    simulates sparse dots from the full mask -- but keeps the full mask
    around too, only for scoring later, never for training."""

    def __init__(self, image_mask_pairs, points_per_class=10, seed=None):
        self.pairs = image_mask_pairs
        self.points_per_class = points_per_class
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img, full_mask = self.pairs[idx]
        sparse_target, point_mask = sample_points_from_mask(
            full_mask, self.points_per_class, rng=self.rng
        )
        img_t = torch.from_numpy(img).permute(2, 0, 1).float()
        full_mask_t = torch.from_numpy(full_mask).long()
        sparse_target_t = torch.from_numpy(sparse_target).long()
        point_mask_t = torch.from_numpy(point_mask).float()
        return img_t, full_mask_t, sparse_target_t, point_mask_t


def compute_miou(pred, target, n_classes, ignore_index=255):
    """The overlapping-circles score, calculated per class and averaged."""
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    valid = target != ignore_index
    pred, target = pred[valid], target[valid]

    ious = {}
    for c in range(n_classes):
        pred_c = pred == c
        target_c = target == c
        union = (pred_c | target_c).sum().item()
        if union == 0:
            continue
        intersection = (pred_c & target_c).sum().item()
        ious[c] = intersection / union

    mean_iou = sum(ious.values()) / len(ious) if ious else 0.0
    return mean_iou, ious

if __name__ == "__main__":
    pred = torch.tensor([[0, 0, 1], [0, 1, 1], [2, 2, 2]])
    target = pred.clone()  # pretend a PERFECT prediction

    miou, per_class = compute_miou(pred, target, n_classes=3)
    print("Per-class IoU:", per_class)
    print("mIoU:", miou, "-- expect 1.0 since prediction is perfect")