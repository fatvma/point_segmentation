import json
from train import train_one_config, _make_synthetic_pairs


def run_points_per_class_experiment(points_values, n_classes=3, epochs=6, pretrained=False):
    # SAME photos used for every run -- only points_per_class changes below
    train_pairs = _make_synthetic_pairs(n_samples=16, seed=100)
    val_pairs = _make_synthetic_pairs(n_samples=8, seed=200)

    results = []
    for p in points_values:
        print(f"\n=== Training with {p} points per class ===")
        _, history = train_one_config(
            train_pairs, val_pairs, n_classes=n_classes, points_per_class=p,
            epochs=epochs, pretrained=pretrained, device="cpu",
        )
        results.append({"points_per_class": p, "final_miou": history[-1]["val_miou"]})
    return results


if __name__ == "__main__":
    results = run_points_per_class_experiment(points_values=[3, 10, 30])

    print("\n================ SUMMARY ================")
    for r in results:
        print(f"points_per_class={r['points_per_class']:>3}  ->  final mIoU={r['final_miou']:.4f}")

    with open("experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results to experiment_results.json")