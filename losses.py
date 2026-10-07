import torch
import torch.nn as nn
import torch.nn.functional as F

class PartialCrossEntropyLoss(nn.Module):
    def forward(self, logits, target, point_mask):
        # normal "how wrong" score, calculated everywhere, nothing skipped yet
        per_pixel_loss = F.cross_entropy(logits, target, reduction="none")

        # multiply by the mask: 1 keeps the score, 0 wipes it out
        point_mask = point_mask.float()
        kept_scores = per_pixel_loss * point_mask

        # average only over the pixels we actually kept
        return kept_scores.sum() / point_mask.sum().clamp_min(1e-7)
    

if __name__ == "__main__":
    torch.manual_seed(0)
    logits = torch.randn(1, 3, 4, 4, requires_grad=True)  # 1 image, 3 classes, 4x4 pixels
    target = torch.randint(0, 3, (1, 4, 4))

    # only label ONE pixel
    point_mask = torch.zeros(1, 4, 4)
    point_mask[0, 1, 2] = 1

    loss_fn = PartialCrossEntropyLoss()
    loss = loss_fn(logits, target, point_mask)
    print("Loss:", loss.item())

    loss.backward()
    print("Gradient at labeled pixel (should be nonzero):", logits.grad[0, :, 1, 2].abs().sum().item())
    print("Gradient at an unlabeled pixel (should be exactly 0):", logits.grad[0, :, 0, 0].abs().sum().item())