import sys
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import json

'''
Standalone script to generate charts and evaluation report from saved model weights for validation set
The main training script (Training ONNX.py) generates saved weights (segmenter.pt and classifier.pt)
This script loads the validation sets, collects predictions and generates charts + report
'''

# add parent directory
sys.path.append(str(Path(__file__).resolve().parent.parent))

from importlib import import_module
training = import_module("Training ONNX")

# pull files for eval
StrokeClassificationDataset = training.StrokeClassificationDataset
StrokeSegmentationDataset = training.StrokeSegmentationDataset
SubsetDataset = training.SubsetDataset
split_indices = training.split_indices
find_png_dir = training.find_png_dir
make_resnet18_classifier = training.make_resnet18_classifier
UNetSmall = training.UNetSmall
unzip_if_needed = training.unzip_if_needed
set_seed = training.set_seed


def main():
    # settings - must match training
    SEED = 42
    VAL_FRAC = 0.15
    DATA_ROOT = Path(__file__).resolve().parent.parent / "extracted_data"
    OUT_DIR = Path(__file__).resolve().parent.parent / "onnx_out"
    CHART_DIR = Path(__file__).resolve().parent / "charts"
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # find image directories
    normal_dir = find_png_dir(DATA_ROOT / "Normal")
    ischemic_dir = find_png_dir(DATA_ROOT / "Ischemic Stroke")
    hemorr_dir = find_png_dir(DATA_ROOT / "Hemorrhagic Stroke")
    ischemic_overlay_dir = find_png_dir(DATA_ROOT / "Ischemic Overlay")
    hemorr_overlay_dir = find_png_dir(DATA_ROOT / "Hemorrhagic Overlay")

    # classifier val set
    from torchvision import transforms
    cls_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    cls_ds = StrokeClassificationDataset(normal_dir, ischemic_dir, hemorr_dir, transform=cls_tf)
    tr_idx, va_idx = split_indices(len(cls_ds), val_frac=VAL_FRAC, seed=SEED)
    cls_val = SubsetDataset(cls_ds, va_idx)
    cls_val_loader = DataLoader(cls_val, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

    # segmenter val set
    seg_ds = StrokeSegmentationDataset(
        ischemic_dir=ischemic_dir,
        hemorr_dir=hemorr_dir,
        ischemic_overlay_dir=ischemic_overlay_dir,
        hemorr_overlay_dir=hemorr_overlay_dir,
        size=(256, 256),
        diff_thresh=25,
        cleanup=True,
    )
    tr_idx_seg, va_idx_seg = split_indices(len(seg_ds), val_frac=VAL_FRAC, seed=SEED)
    seg_val = SubsetDataset(seg_ds, va_idx_seg)
    seg_val_loader = DataLoader(seg_val, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

    print(f"Classifier val set: {len(cls_val)} images")
    print(f"Segmenter val set: {len(seg_val)} pairs")
    
    # load classifier
    cls_model = make_resnet18_classifier(num_classes=3, pretrained=False).to(device)
    cls_model.load_state_dict(torch.load(OUT_DIR / "classifier.pt", map_location=device))
    cls_model.eval()
    print("Classifier loaded from classifier.pt")

    # load segmenter
    seg_model = UNetSmall(in_ch=1, base=64).to(device)
    seg_model.load_state_dict(torch.load(OUT_DIR / "segmenter.pt", map_location=device))
    seg_model.eval()
    print("Segmenter loaded from segmenter.pt")
    
    # collect classifier predictions
    labels = ["Normal", "Ischemic", "Hemorrhagic"]
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for x, y in cls_val_loader:
            x = x.to(device)
            logits = cls_model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y.numpy())
            all_probs.extend(probs)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    print(f"Collected {len(all_preds)} classifier predictions")
    
    # confusion matrix heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        all_targets, all_preds,
        display_labels=labels,
        cmap='Blues',
        ax=ax
    )
    ax.set_title('Classification confusion matrix')
    plt.tight_layout()
    plt.savefig(CHART_DIR / "confusion_matrix.png", dpi=150)
    plt.close()
    print(f"Saved: confusion_matrix.png")
    
    # per-class ROC curves
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#1f77b4', '#ff7f0e', '#d62728']
    for i, (label, color) in enumerate(zip(labels, colors)):
        binary_targets = (all_targets == i).astype(int)
        fpr, tpr, _ = roc_curve(binary_targets, all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f'{label} (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title('Per-class ROC curves (one-vs-rest)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(CHART_DIR / "roc_curves.png", dpi=150)
    plt.close()
    print(f"Saved: roc_curves.png")
    
    # training curves from saved epoch history
    history_path = OUT_DIR / "training_history.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        # classifier accuracy
        cls_epochs = list(range(1, len(history["cls"]["train_acc"]) + 1))
        ax1.plot(cls_epochs, history["cls"]["train_acc"], 'b-o', label='Train', markersize=4)
        ax1.plot(cls_epochs, history["cls"]["val_acc"], 'r-o', label='Val', markersize=4)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Classifier: accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # classifier loss
        ax2.plot(cls_epochs, history["cls"]["train_loss"], 'b-o', label='Train loss', markersize=4)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Classifier: training loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # segmenter loss
        seg_epochs = list(range(1, len(history["seg"]["train_loss"]) + 1))
        ax3.plot(seg_epochs, history["seg"]["train_loss"], 'b-o', label='Train', markersize=3)
        ax3.plot(seg_epochs, history["seg"]["val_loss"], 'r-o', label='Val', markersize=3)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss (BCE + Dice)')
        ax3.set_title('Segmenter: loss')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # segmenter dice
        ax4.plot(seg_epochs, history["seg"]["val_dice"], 'g-o', label='Val Dice', markersize=3)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Dice score')
        ax4.set_title('Segmenter: validation Dice')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(CHART_DIR / "training_curves.png", dpi=150)
        plt.close()
        print(f"Saved: training_curves.png")
    else:
        print("No training_history.json found — skipping training curves")
    
    print(f"\nAll charts saved to: {CHART_DIR.resolve()}")

    # evaluation report
    cm = confusion_matrix(all_targets, all_preds)
    sensitivities = cm.diagonal() / cm.sum(axis=1)
    
    # compute AUCs
    aucs = []
    for i in range(3):
        binary_targets = (all_targets == i).astype(int)
        fpr, tpr, _ = roc_curve(binary_targets, all_probs[:, i])
        aucs.append(auc(fpr, tpr))

    # segmenter metrics
    seg_TP = seg_FP = seg_FN = seg_TN = 0
    seg_dices = []
    with torch.no_grad():
        for x, mask in seg_val_loader:
            x = x.to(device)
            mask = mask.to(device)
            logits = seg_model(x)
            pred = (torch.sigmoid(logits).squeeze(1) > 0.5).float()
            seg_TP += (pred * mask).sum().item()
            seg_FP += (pred * (1 - mask)).sum().item()
            seg_FN += ((1 - pred) * mask).sum().item()
            seg_TN += ((1 - pred) * (1 - mask)).sum().item()
            inter = (pred * mask).sum(dim=(1,2))
            union = pred.sum(dim=(1,2)) + mask.sum(dim=(1,2))
            dice = (2 * inter + 1e-6) / (union + 1e-6)
            seg_dices.extend(dice.cpu().tolist())

    seg_precision = seg_TP / (seg_TP + seg_FP + 1e-7)
    seg_recall = seg_TP / (seg_TP + seg_FN + 1e-7)
    seg_f1 = 2 * seg_precision * seg_recall / (seg_precision + seg_recall + 1e-7)
    seg_dice_mean = float(np.mean(seg_dices))

    report = f"""========================================
  STROKE DETECTION EVALUATION REPORT
========================================

CLASSIFIER (ResNet18, 3-class)
------------------------------
Overall Accuracy: {(all_preds == all_targets).mean():.4f}

Per-class Sensitivity (Recall):
  Normal:       {sensitivities[0]:.4f} ({int(cm[0,0])}/{int(cm[0].sum())})
  Ischemic:     {sensitivities[1]:.4f} ({int(cm[1,1])}/{int(cm[1].sum())})
  Hemorrhagic:  {sensitivities[2]:.4f} ({int(cm[2,2])}/{int(cm[2].sum())})

Per-class AUC (one-vs-rest):
  Normal:       {aucs[0]:.4f}
  Ischemic:     {aucs[1]:.4f}
  Hemorrhagic:  {aucs[2]:.4f}
  Macro AUC:    {np.mean(aucs):.4f}

Confusion Matrix:
              Pred Normal  Pred Ischemic  Pred Hemorrhagic
  Normal        {cm[0,0]:>6}         {cm[0,1]:>6}            {cm[0,2]:>6}
  Ischemic      {cm[1,0]:>6}         {cm[1,1]:>6}            {cm[1,2]:>6}
  Hemorrhagic   {cm[2,0]:>6}         {cm[2,1]:>6}            {cm[2,2]:>6}

Strokes missed as Normal: {int(cm[1,0] + cm[2,0])} / {int(cm[1].sum() + cm[2].sum())}

SEGMENTER (UNet, binary mask)
------------------------------
Mean Dice:       {seg_dice_mean:.4f}
Mean IoU:        {float(np.mean([(2*d)/(1+d) for d in seg_dices])):.4f}
Pixel Precision: {seg_precision:.4f}
Pixel Recall:    {seg_recall:.4f}
Pixel F1:        {seg_f1:.4f}

========================================
"""

    print(report)
    
    report_path = CHART_DIR / "evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved: evaluation_report.txt")
if __name__ == "__main__":
    main()
    