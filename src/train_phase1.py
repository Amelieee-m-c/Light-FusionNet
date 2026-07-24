"""
Phase 1: fine-tune EfficientNetV2B0(+SE) end-to-end for maize leaf disease
classification (Section II-E). Adam lr=1e-4, label smoothing 0.1, up to 50
epochs, early stopping (patience=10, monitors val accuracy), class weights,
best-val checkpoint restored at the end.

Preprocessing: resize 256x256, "EfficientNetV2 preprocessing" -- Keras'
efficientnet_v2.preprocess_input is a rescale to [-1,1] (x/127.5 - 1), which
we replicate as ToTensor() + Normalize(mean=0.5, std=0.5) per channel.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.utils.class_weight import compute_class_weight

from model import LightETBackbone, count_params


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def build_transforms(img_size: int):
    normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        normalize,
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf, eval_tf


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss, total_correct, total_n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            total_correct += (out.argmax(1) == y).sum().item()
            total_n += x.size(0)
    return total_loss / total_n, total_correct / total_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="dir with train/val/test subfolders")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--no_se", action="store_true", help="ablation: disable the SE block")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    train_tf, eval_tf = build_transforms(args.img_size)

    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=eval_tf)
    test_ds = datasets.ImageFolder(data_dir / "test", transform=eval_tf)
    assert train_ds.classes == val_ds.classes == test_ds.classes
    class_names = train_ds.classes
    num_classes = len(class_names)
    print(f"classes ({num_classes}): {class_names}")
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    persistent = args.num_workers > 0
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True, persistent_workers=persistent)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True, persistent_workers=persistent)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True, persistent_workers=persistent)

    model = LightETBackbone(num_classes=num_classes).to(device)
    if args.no_se:
        model.se = nn.Identity().to(device)  # ablation: bypass SE gating
    trainable, non_trainable, total = count_params(model)
    print(f"trainable={trainable} non_trainable(buffers not counted here)={non_trainable} total={total}")

    class_weights = compute_class_weight("balanced", classes=np.arange(num_classes),
                                          y=np.array(train_ds.targets))
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)
    print("class weights:", class_weights)

    criterion = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = -1.0
    best_state = None
    epochs_no_improve = 0
    history = []

    for epoch in range(args.epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                         "val_loss": val_loss, "val_acc": val_acc})
        print(f"epoch {epoch+1}/{args.epochs}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"early stopping at epoch {epoch+1} (best val_acc={best_val_acc:.4f})")
                break

    model.load_state_dict(best_state)
    torch.save(best_state, out_dir / "best_backbone.pt")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(out_dir / "class_names.json", "w") as f:
        json.dump(class_names, f, indent=2)

    test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer, device, train=False)
    print(f"\nphase-1 CNN test accuracy (softmax head, before ensemble): {test_acc:.4f}")
    with open(out_dir / "phase1_test_acc.json", "w") as f:
        json.dump({"test_acc": test_acc, "best_val_acc": best_val_acc, "stopped_epoch": len(history)}, f, indent=2)

    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
