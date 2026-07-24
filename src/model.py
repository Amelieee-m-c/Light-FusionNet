"""
LightET-FusionNet reproduction (PyTorch), phase 1 (CNN + SE backbone).

Reference: Shahzad, Amjad, Mushtaq, Chughtai, "LightET-FusionNet as a
Lightweight Deep Ensemble Model for Scalable Maize Disease Classification",
IEEE Access, 2026. No official code repo found; independent clean-room
reimplementation from Section II-D (eqs. 1-7), Figure 2, Section II-E.

Architecture:
  EfficientNetV2B0 (ImageNet-pretrained, via timm's `tf_efficientnetv2_b0` --
  a direct port of the Keras/TF weights the paper actually used, unlike
  torchvision's differently-scaled efficientnet_v2_s/m/l) -> GAP (1280-d)
  -> custom SE block (channel gating on the *pooled* vector, reduction 1:8)
  -> Dense(256, ReLU) -> Dropout -> Dense(128) -> [phase-1 only] Dense(num_classes, softmax)

For phase 2 (see `extract_features.py` / `train_ensemble.py`), the
penultimate Dense(128) layer's output is taken as the feature embedding fed
to ExtraTrees (and other sklearn classifiers).

Known paper ambiguities, resolved here (documented, not silently guessed):

  1. **SE block operand**: Section II-D's prose says the SE block is
     "manually integrated after the GAP layer" and operates on "the pooled
     descriptor" (the 1x1xC vector) -- but eq. (4) writes F^SE = F (dot) s
     with F in R^(HxWxC), i.e. the *pre-GAP spatial* feature map. These
     contradict each other. We follow the prose + Figure 2 (which draws the
     SE block strictly after the "Avg Pooling" box, before the Dense(256)
     layer): the SE gate is computed from and applied to the already-pooled
     1280-d vector, not the spatial map. This is also the simpler,
     `cheaper reading and matches the paper's own emphasis on being
     "lightweight."
  2. **Feature-extraction layer for phase 2**: the text says features come
     from "the third-to-last dense layer," but counting the described stack
     backwards (softmax(num_classes) -> Dense(128) -> Dropout -> Dense(256))
     puts the "third-to-last" at Dropout, not Dense(128) -- yet Figure 2
     explicitly labels the ExtraTrees input as "128 Features." We trust
     Figure 2's concrete number and take features from the Dense(128) layer.
  3. Activation on the Dense(128) layer isn't stated; we use ReLU, matching
     the Dense(256) layer immediately before it.
"""
import torch
import torch.nn as nn
import timm


class SqueezeExcite1D(nn.Module):
    """SE gating on an already-pooled (B, C) descriptor -- reduction 1:8,
    ReLU then sigmoid, per eq. (3): SE = sigmoid(W2 . ReLU(W1 . z))."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = channels // reduction
        self.w1 = nn.Linear(channels, hidden, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.w2 = nn.Linear(hidden, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        s = self.sigmoid(self.w2(self.relu(self.w1(z))))
        return z * s


class LightETBackbone(nn.Module):
    """Phase-1 CNN: EfficientNetV2B0 -> GAP -> SE -> Dense(256) -> Dropout
    -> Dense(128) -> [training-only] Dense(num_classes)."""

    def __init__(self, num_classes: int, pretrained: bool = True,
                 se_reduction: int = 8, dropout: float = 0.3):
        super().__init__()
        self.backbone = timm.create_model(
            "tf_efficientnetv2_b0", pretrained=pretrained, num_classes=0  # -> pooled (B, 1280) features
        )
        feat_dim = self.backbone.num_features  # 1280
        self.se = SqueezeExcite1D(feat_dim, reduction=se_reduction)

        self.dense256 = nn.Linear(feat_dim, 256)
        self.act256 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.dense128 = nn.Linear(256, 128)
        self.act128 = nn.ReLU(inplace=True)

        self.classifier = nn.Linear(128, num_classes)  # phase-1 training head only

    def embed(self, x):
        """Returns the 128-d feature embedding (for phase-2 ensemble use)."""
        z = self.backbone(x)         # (B, 1280) pooled
        z = self.se(z)                # (B, 1280) SE-recalibrated
        h = self.act256(self.dense256(z))
        h = self.dropout(h)
        h = self.act128(self.dense128(h))
        return h  # (B, 128)

    def forward(self, x):
        h = self.embed(x)
        return self.classifier(h)


def count_params(model: nn.Module):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total - trainable, total  # trainable, non-trainable, total


if __name__ == "__main__":
    m = LightETBackbone(num_classes=4)
    x = torch.randn(2, 3, 256, 256)
    y = m(x)
    emb = m.embed(x)
    print("logits shape:", y.shape, "embedding shape:", emb.shape)
    trainable, non_trainable, total = count_params(m)
    print(f"trainable={trainable}  non_trainable={non_trainable}  total={total}")
