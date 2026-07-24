"""
Builds a 70/15/15 stratified train/val/test split (paper's Section III-B),
reusing the maize dataset already downloaded for DenseViT_repro (same
underlying "Corn or Maize Leaf Disease Dataset" -- this paper cites
sahityamamillapalli's re-upload, DenseViT_repro used smaranjitghose's; both
mirror the same 4188-image, 4-class dataset, confirmed by identical file
size and class distribution).

Supports the paper's 3-independent-seeds protocol via --seed.
"""
import argparse
import random
import shutil
from pathlib import Path

SOURCE = Path(r"E:\plant_disease\DenseViT_repro\data\raw\corn\data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--train_frac", type=float, default=0.70)
    ap.add_argument("--val_frac", type=float, default=0.15)
    args = ap.parse_args()

    random.seed(args.seed)
    out = Path(args.output)

    classes = {d.name: [f for f in d.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
               for d in SOURCE.iterdir() if d.is_dir()}

    print(f"classes: {len(classes)}")
    totals = {"train": 0, "val": 0, "test": 0}
    for c, files in sorted(classes.items()):
        random.shuffle(files)
        n = len(files)
        n_train = round(n * args.train_frac)
        n_val = round(n * args.val_frac)
        splits = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:],
        }
        for split, split_files in splits.items():
            (out / split / c).mkdir(parents=True, exist_ok=True)
            for f in split_files:
                shutil.copy2(f, out / split / c / f.name)
            totals[split] += len(split_files)
        print(f"  {c:20s} total={n:5d} train={len(splits['train']):5d} val={len(splits['val']):5d} test={len(splits['test']):5d}")

    print(f"\nTOTAL train={totals['train']} val={totals['val']} test={totals['test']} (grand total={sum(totals.values())})")
    print(f"saved to: {out}")


if __name__ == "__main__":
    main()
