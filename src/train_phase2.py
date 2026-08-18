"""
Phase 2: extract 128-d embeddings from the phase-1-trained backbone (Dense
layer immediately before its softmax head -- see model.py docstring point 2
for why), StandardScale them, then fit + evaluate several classical ML
classifiers on top, matching the paper's Table 4 comparison. ExtraTrees is
the paper's selected "LightET-FusionNet" configuration.

Classifier configs per Section II-E (grid-searched values not searched here
-- we pick the paper's stated defaults directly, noted where a choice like
"None or 20" wasn't resolved to one value in the paper text):
  - SVM: RBF kernel, C=1.0, gamma='scale'
  - GB: 200 estimators, lr=0.1, max_depth=3
  - XGBoost: 300 trees, max_depth=6, lr=0.1, subsample=0.9
  - LR: L2, C=1.0, max_iter=2000
  - LightGBM: 500 estimators, 31 leaves, lr=0.1
  - ExtraTrees (LightET-FusionNet): 200 estimators, max_depth=20 (paper's
    GridSearchCV considered "None or 20" -- originally left at the sklearn
    default None since the paper text didn't say which one won; confirmed
    to be 20), max_features='sqrt', class_weight='balanced'
  - VotingEnsemble: soft-voting over {LR, SVM, GB} (paper names "Voting
    Ensemble" as one of the compared classifiers without specifying its
    member models; we chose a reasonable 3-member composition)
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from model import LightETBackbone


def build_eval_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()
    feats, labels = [], []
    for x, y in loader:
        x = x.to(device)
        h = model.embed(x)
        feats.append(h.cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def predict_hard_voting(clf, X):
    """True hard voting across an ExtraTreesClassifier's internal trees: each
    tree casts one vote for its predicted class, majority wins -- distinct
    from sklearn's built-in .predict(), which averages each tree's predicted
    class *probabilities* (soft voting) and argmaxes that. The paper is
    explicit that ET "follows a hard-voting strategy in which all decision
    trees contribute equally... without assigning explicit weights," which
    sklearn's default does not actually implement."""
    tree_preds = np.stack([tree.predict(X) for tree in clf.estimators_], axis=0).astype(int)  # (n_trees, n_samples)
    n_classes = int(tree_preds.max()) + 1
    # one-hot count votes per class per sample, then take the argmax (majority)
    one_hot = np.eye(n_classes, dtype=int)[tree_preds]  # (n_trees, n_samples, n_classes)
    vote_counts = one_hot.sum(axis=0)  # (n_samples, n_classes)
    return vote_counts.argmax(axis=1)


def build_classifiers(seed: int):
    return {
        "GradientBoosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=seed),
        "LogisticRegression": LogisticRegression(C=1.0, max_iter=2000, penalty="l2", random_state=seed),
        "SVM": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=seed),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.9,
                                  random_state=seed, eval_metric="mlogloss"),
        "LightGBM": LGBMClassifier(n_estimators=500, num_leaves=31, learning_rate=0.1, random_state=seed, verbosity=-1),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=200, max_depth=20, max_features="sqrt",
                                            class_weight="balanced", random_state=seed),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--backbone_dir", required=True, help="dir with best_backbone.pt + class_names.json from train_phase1.py")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    backbone_dir = Path(args.backbone_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(backbone_dir / "class_names.json") as f:
        class_names = json.load(f)
    num_classes = len(class_names)

    model = LightETBackbone(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(backbone_dir / "best_backbone.pt", map_location=device))
    model.eval()

    tf = build_eval_transform(args.img_size)
    train_ds = datasets.ImageFolder(data_dir / "train", transform=tf)
    test_ds = datasets.ImageFolder(data_dir / "test", transform=tf)
    assert train_ds.classes == test_ds.classes == class_names

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print("extracting features...")
    X_train, y_train = extract_features(model, train_loader, device)
    X_test, y_test = extract_features(model, test_loader, device)
    print(f"train features: {X_train.shape}  test features: {X_test.shape}")

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    classifiers = build_classifiers(args.seed)
    voting = VotingClassifier(
        estimators=[("lr", classifiers["LogisticRegression"]),
                    ("svm", classifiers["SVM"]),
                    ("gb", classifiers["GradientBoosting"])],
        voting="soft",
    )
    classifiers["VotingEnsemble"] = voting

    results = {}
    for name, clf in classifiers.items():
        print(f"\n=== {name} ===")
        clf.fit(X_train_s, y_train)
        if name == "ExtraTrees":
            y_pred = predict_hard_voting(clf, X_test_s)
        else:
            y_pred = clf.predict(X_test_s)
        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=class_names, digits=4, zero_division=0)
        print(f"accuracy={acc:.4f} precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")
        results[name] = {
            "accuracy": acc, "precision_macro": precision, "recall_macro": recall, "f1_macro": f1,
            "confusion_matrix": cm.tolist(), "classification_report": report,
        }

    with open(out_dir / "phase2_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved: {out_dir / 'phase2_results.json'}")

    # persist the fitted scaler + ExtraTrees classifier (the paper's namesake
    # "LightET-FusionNet" model) so the full backbone->embedding->ET pipeline
    # is actually loadable/usable, not just the CNN backbone half of it.
    joblib.dump(scaler, out_dir / "phase2_scaler.joblib")
    joblib.dump(classifiers["ExtraTrees"], out_dir / "phase2_extratrees.joblib")
    print(f"saved: {out_dir / 'phase2_scaler.joblib'}, {out_dir / 'phase2_extratrees.joblib'}")
    print("\n=== Summary (LightET-FusionNet = ExtraTrees row) ===")
    for name, r in results.items():
        print(f"{name:20s} acc={r['accuracy']*100:.2f}  prec={r['precision_macro']*100:.2f}  "
              f"rec={r['recall_macro']*100:.2f}  f1={r['f1_macro']*100:.2f}")


if __name__ == "__main__":
    main()
