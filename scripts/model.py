"""MobileNetV3-Large builder, two-stage freeze control, checkpoint I/O.

Stage 1 trains the head on a frozen backbone; stage 2 additionally unfreezes
features[UNFREEZE_FROM:]. BatchNorm running statistics are held fixed wherever
the surrounding weights are frozen -- see _install_bn_guard.
"""
import types
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

import config as C

UNFREEZE_FROM = 12          # stage 2 unfreezes features[12:] + classifier


# ── Build ─────────────────────────────────────────────────────────────────
def build_model(num_classes=C.NUM_CLASSES, pretrained=True, dropout=0.2):
    """ImageNet-pretrained MobileNetV3-Large with the final layer resized.

    Only classifier[3] is replaced; the 960->1280 projection is pretrained
    feature machinery, not an ImageNet label mapping, and is kept.
    """
    weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.mobilenet_v3_large(weights=weights)

    in_f = m.classifier[3].in_features          # 1280
    m.classifier[3] = nn.Linear(in_f, num_classes)
    m.classifier[2].p = dropout
    return m


# ── Stage control ─────────────────────────────────────────────────────────
def set_stage(model, stage, unfreeze_from=UNFREEZE_FROM):
    """stage 1 = head only (backbone frozen). stage 2 = head + late blocks."""
    if stage not in (1, 2):
        raise ValueError("stage must be 1 or 2")

    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True

    if stage == 2:
        for blk in model.features[unfreeze_from:]:
            for p in blk.parameters():
                p.requires_grad = True

    model._stage = stage
    _install_bn_guard(model)
    return model


def _install_bn_guard(model):
    """Override .train() so frozen BatchNorm layers never update running stats.

    requires_grad=False stops gradients but NOT the running mean/var update,
    which happens on every forward pass in training mode. Left unhandled, a
    'frozen' backbone drifts its normalisation statistics away from the weights
    that were tuned against them -- silent, and it only shows up as validation
    metrics that are inexplicably worse than training.

    Wired into .train() rather than the training loop because .train() is called
    every epoch and this must hold every time.
    """
    if getattr(model, "_bn_guard", False):
        return model

    original_train = model.train

    def train(self, mode=True):
        original_train(mode)
        if mode:
            for i, blk in enumerate(self.features):
                if i < UNFREEZE_FROM or self._stage == 1:
                    for m in blk.modules():
                        if isinstance(m, nn.modules.batchnorm._BatchNorm):
                            m.eval()
        return self

    model.train = types.MethodType(train, model)
    model._bn_guard = True
    return model


def count_params(model):
    """(trainable, frozen) parameter counts."""
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fz = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return tr, fz


# ── Optimizer groups ──────────────────────────────────────────────────────
def param_groups(model, stage, lr_head=C.LR_HEAD, lr_backbone=C.LR_BACKBONE):
    """Trainable params only, split so stage 2 moves the backbone slower.

    Passing frozen tensors to the optimizer allocates state for parameters that
    never update, so this filters on requires_grad.
    """
    head = [p for p in model.classifier.parameters() if p.requires_grad]
    back = [p for p in model.features.parameters() if p.requires_grad]

    groups = [{"params": head, "lr": lr_head if stage == 1 else lr_head / 10,
               "name": "head"}]
    if back:
        groups.append({"params": back, "lr": lr_backbone, "name": "backbone"})
    return groups


# ── Checkpoints ───────────────────────────────────────────────────────────
def save_checkpoint(path, model, *, fold, stage, epoch, metrics, aug_mode=None):
    """Weights plus everything needed to reload and serve them correctly.

    class_to_idx and aug_mode live inside the file deliberately: the serving
    API must not depend on a separate mapping that can drift out of sync.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict":    model.state_dict(),
        "arch":          "mobilenet_v3_large",
        "class_to_idx":  C.CLASS_TO_IDX,
        "img_size":      C.IMG_SIZE,
        "aug_mode":      aug_mode or C.AUG_MODE,
        "unfreeze_from": UNFREEZE_FROM,
        "fold":          fold,
        "stage":         stage,
        "epoch":         epoch,
        "metrics":       metrics,
        "seed":          C.SEED,
    }, path)
    return path


def load_checkpoint(path, device=None):
    """Rebuild the model (no weight download) and restore. Returns (model, meta)."""
    ck = torch.load(path, map_location=device or "cpu", weights_only=False)
    if ck["class_to_idx"] != C.CLASS_TO_IDX:
        raise ValueError(f"class mapping mismatch: {ck['class_to_idx']}")

    m = build_model(num_classes=len(ck["class_to_idx"]), pretrained=False)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    if device:
        m.to(device)
    return m, {k: v for k, v in ck.items() if k != "state_dict"}
