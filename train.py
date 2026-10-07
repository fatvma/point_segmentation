import numpy as np
import torch
from torch.utils.data import DataLoader

from model import ResNetUNet
from losses import PartialCrossEntropyLoss
from dataset import PointSupervisedSegDataset, compute_miou


def train_one_config(train_pairs, val_pairs, n_classes, points_per_class,
                      epochs=5, lr=1e-3, pretrained=True, device="cpu"):
    train_ds = PointSupervisedSegDataset(train_pairs, points_per_class=points_per_class, seed=0)
    val_ds = PointSupervisedSegDataset(val_pairs, points_per_class=points_per_class, seed=1)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)

    model = ResNetUNet(n_classes=n_classes, pretrained=pretrained).to(device)
    criterion = PartialCrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    for epoch in range(epochs):
        # ---- TRAINING: show the model examples, adjust it ----
        model.train()
        total_loss = 0.0
        for img, full_mask, sparse_target, point_mask in train_loader:
            optimizer.zero_grad()                     # clear old adjustment signal
            logits = model(img)                       # model's guess
            loss = criterion(logits, sparse_target, point_mask)  # how wrong, using only the dots
            loss.backward()                            # figure out which direction to adjust
            optimizer.step()                           # actually adjust the model
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        # ---- VALIDATION: check progress using the FULL mask (never used for training) ----
        model.eval()
        all_miou = []
        with torch.no_grad():
            for img, full_mask, sparse_target, point_mask in val_loader:
                logits = model(img)
                pred = logits.argmax(dim=1)  # pick the highest-scoring class per pixel
                for b in range(pred.shape[0]):
                    miou, _ = compute_miou(pred[b], full_mask[b], n_classes)
                    all_miou.append(miou)
        val_miou = float(np.mean(all_miou))

        history.append({"epoch": epoch, "train_loss": avg_loss, "val_miou": val_miou})
        print(f"  epoch {epoch}: loss={avg_loss:.4f}  val_mIoU={val_miou:.4f}")

    return model, history

def _make_synthetic_pairs(n_samples=8, size=64, n_classes=3, seed=0):
    rng = np.random.default_rng(seed)
    pairs = []
    for _ in range(n_samples):
        mask = np.zeros((size, size), dtype=np.int64)
        third = size // n_classes
        for c in range(n_classes):
            mask[c*third:(c+1)*third, :] = c
        img = np.zeros((size, size, 3), dtype=np.float32)
        colors = rng.random((n_classes, 3))
        for c in range(n_classes):
            img[mask == c] = colors[c]
        pairs.append((img, mask))
    return pairs


if __name__ == "__main__":
    print("Running a quick pipeline smoke test on fake data...")
    train_pairs = _make_synthetic_pairs(n_samples=8, seed=1)
    val_pairs = _make_synthetic_pairs(n_samples=4, seed=2)
    model, history = train_one_config(
        train_pairs, val_pairs, n_classes=3, points_per_class=10,
        epochs=3, pretrained=False, device="cpu",
    )
    print("Smoke test done -- if you see loss numbers and mIoU numbers above with no errors, the full pipeline works.")