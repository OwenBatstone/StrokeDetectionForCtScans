import sys
import time
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    classification_report, cohen_kappa_score,
    precision_score, recall_score, f1_score
)
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
    

    # CLASSIFIER: collect predictions
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


    # Confusion matrix heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        all_targets, all_preds,
        display_labels=labels,
        cmap='Blues',
        ax=ax
    )
    ax.set_title('Classification Confusion Matrix')
    plt.tight_layout()
    plt.savefig(CHART_DIR / "confusion_matrix.png", dpi=150)
    plt.close()
    print(f"Saved: confusion_matrix.png")


    # Normalized confusion matrix (percentages)
    cm = confusion_matrix(all_targets, all_preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Normalized Confusion Matrix (%)')
    for i in range(3):
        for j in range(3):
            color = 'white' if cm_norm[i, j] > 0.5 else 'black'
            ax.text(j, i, f'{cm_norm[i,j]:.1%}', ha='center', va='center', color=color, fontsize=14)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(CHART_DIR / "confusion_matrix_normalized.png", dpi=150)
    plt.close()
    print(f"Saved: confusion_matrix_normalized.png")
    

    # Per-class ROC curves
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#1f77b4', '#ff7f0e', '#d62728']
    aucs = []
    for i, (label, color) in enumerate(zip(labels, colors)):
        binary_targets = (all_targets == i).astype(int)
        fpr, tpr, _ = roc_curve(binary_targets, all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, color=color, lw=2, label=f'{label} (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Per-class ROC Curves (One-vs-Rest)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(CHART_DIR / "roc_curves.png", dpi=150)
    plt.close()
    print(f"Saved: roc_curves.png")


    # Precision-Recall curves
    fig, ax = plt.subplots(figsize=(8, 6))
    avg_precisions = []
    for i, (label, color) in enumerate(zip(labels, colors)):
        binary_targets = (all_targets == i).astype(int)
        prec_vals, rec_vals, _ = precision_recall_curve(binary_targets, all_probs[:, i])
        ap = average_precision_score(binary_targets, all_probs[:, i])
        avg_precisions.append(ap)
        ax.plot(rec_vals, prec_vals, color=color, lw=2, label=f'{label} (AP = {ap:.3f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Per-class Precision-Recall Curves')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(CHART_DIR / "precision_recall_curves.png", dpi=150)
    plt.close()
    print(f"Saved: precision_recall_curves.png")


    # Confidence distribution histogram
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, (label, ax) in enumerate(zip(labels, axes)):
        correct_mask = (all_preds == i) & (all_targets == i)
        incorrect_mask = (all_preds == i) & (all_targets != i)
        if correct_mask.any():
            ax.hist(all_probs[correct_mask, i], bins=20, alpha=0.7, color='green', label='Correct', range=(0, 1))
        if incorrect_mask.any():
            ax.hist(all_probs[incorrect_mask, i], bins=20, alpha=0.7, color='red', label='Incorrect', range=(0, 1))
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Count')
        ax.set_title(f'{label}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.suptitle('Classifier Confidence Distribution (Correct vs Incorrect)', fontsize=14)
    plt.tight_layout()
    plt.savefig(CHART_DIR / "confidence_distribution.png", dpi=150)
    plt.close()
    print(f"Saved: confidence_distribution.png")


    # Per-class metrics bar chart
    cls_precision = precision_score(all_targets, all_preds, average=None, zero_division=0)
    cls_recall = recall_score(all_targets, all_preds, average=None, zero_division=0)
    cls_f1 = f1_score(all_targets, all_preds, average=None, zero_division=0)

    x_pos = np.arange(3)
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x_pos - width, cls_precision, width, label='Precision', color='#2196F3')
    ax.bar(x_pos, cls_recall, width, label='Recall', color='#FF9800')
    ax.bar(x_pos + width, cls_f1, width, label='F1 Score', color='#4CAF50')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Score')
    ax.set_title('Per-class Precision, Recall, and F1 Score')
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    for i in range(3):
        ax.text(i - width, cls_precision[i] + 0.02, f'{cls_precision[i]:.2f}', ha='center', fontsize=9)
        ax.text(i, cls_recall[i] + 0.02, f'{cls_recall[i]:.2f}', ha='center', fontsize=9)
        ax.text(i + width, cls_f1[i] + 0.02, f'{cls_f1[i]:.2f}', ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(CHART_DIR / "per_class_metrics.png", dpi=150)
    plt.close()
    print(f"Saved: per_class_metrics.png")
    

    # Training curves from saved epoch history
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
        ax1.set_title('Classifier: Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # classifier loss
        ax2.plot(cls_epochs, history["cls"]["train_loss"], 'b-o', label='Train loss', markersize=4)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Classifier: Training Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # segmenter loss
        seg_epochs = list(range(1, len(history["seg"]["train_loss"]) + 1))
        ax3.plot(seg_epochs, history["seg"]["train_loss"], 'b-o', label='Train', markersize=3)
        ax3.plot(seg_epochs, history["seg"]["val_loss"], 'r-o', label='Val', markersize=3)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss (BCE + Dice)')
        ax3.set_title('Segmenter: Loss')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # segmenter dice
        ax4.plot(seg_epochs, history["seg"]["val_dice"], 'g-o', label='Val Dice', markersize=3)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Dice Score')
        ax4.set_title('Segmenter: Validation Dice')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(CHART_DIR / "training_curves.png", dpi=150)
        plt.close()
        print(f"Saved: training_curves.png")
    else:
        print("No training_history.json found — skipping training curves")


    # SEGMENTER: collect metrics
    seg_TP = seg_FP = seg_FN = seg_TN = 0
    seg_dices = []
    seg_ious = []

    with torch.no_grad():
        for x, mask in seg_val_loader:
            x = x.to(device)
            mask = mask.to(device)
            logits = seg_model(x)
            pred = (torch.sigmoid(logits).squeeze(1) > 0.5).float()

            tp = (pred * mask).sum().item()
            fp = (pred * (1 - mask)).sum().item()
            fn = ((1 - pred) * mask).sum().item()
            tn = ((1 - pred) * (1 - mask)).sum().item()
            seg_TP += tp
            seg_FP += fp
            seg_FN += fn
            seg_TN += tn

            # per-image dice and IoU
            inter = (pred * mask).sum(dim=(1, 2))
            union = pred.sum(dim=(1, 2)) + mask.sum(dim=(1, 2))
            dice = (2 * inter + 1e-6) / (union + 1e-6)
            iou = (inter + 1e-6) / (union - inter + 1e-6)
            seg_dices.extend(dice.cpu().tolist())
            seg_ious.extend(iou.cpu().tolist())

    seg_precision = seg_TP / (seg_TP + seg_FP + 1e-7)
    seg_recall = seg_TP / (seg_TP + seg_FN + 1e-7)
    seg_f1 = 2 * seg_precision * seg_recall / (seg_precision + seg_recall + 1e-7)
    seg_pixel_acc = (seg_TP + seg_TN) / (seg_TP + seg_TN + seg_FP + seg_FN + 1e-7)
    seg_specificity = seg_TN / (seg_TN + seg_FP + 1e-7)
    seg_dice_mean = float(np.mean(seg_dices))
    seg_dice_std = float(np.std(seg_dices))
    seg_iou_mean = float(np.mean(seg_ious))
    seg_iou_std = float(np.std(seg_ious))


    # Segmenter Dice score distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.hist(seg_dices, bins=30, color='#4CAF50', alpha=0.8, edgecolor='black')
    ax1.axvline(seg_dice_mean, color='red', linestyle='--', lw=2, label=f'Mean = {seg_dice_mean:.3f}')
    ax1.set_xlabel('Dice Score')
    ax1.set_ylabel('Count')
    ax1.set_title('Segmenter: Dice Score Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.hist(seg_ious, bins=30, color='#2196F3', alpha=0.8, edgecolor='black')
    ax2.axvline(seg_iou_mean, color='red', linestyle='--', lw=2, label=f'Mean = {seg_iou_mean:.3f}')
    ax2.set_xlabel('IoU Score')
    ax2.set_ylabel('Count')
    ax2.set_title('Segmenter: IoU Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(CHART_DIR / "segmenter_score_distribution.png", dpi=150)
    plt.close()
    print(f"Saved: segmenter_score_distribution.png")


    # Segmenter threshold analysis
    thresholds = np.arange(0.1, 0.95, 0.05)
    thresh_dices = []
    thresh_precisions = []
    thresh_recalls = []

    with torch.no_grad():
        # collect all logits and masks first
        all_seg_logits = []
        all_seg_masks = []
        for x, mask in seg_val_loader:
            x = x.to(device)
            mask = mask.to(device)
            logits = seg_model(x)
            all_seg_logits.append(torch.sigmoid(logits).squeeze(1))
            all_seg_masks.append(mask)
        all_seg_probs = torch.cat(all_seg_logits, dim=0)
        all_seg_masks = torch.cat(all_seg_masks, dim=0)

        for t in thresholds:
            pred = (all_seg_probs > t).float()
            tp = (pred * all_seg_masks).sum().item()
            fp = (pred * (1 - all_seg_masks)).sum().item()
            fn = ((1 - pred) * all_seg_masks).sum().item()
            p = tp / (tp + fp + 1e-7)
            r = tp / (tp + fn + 1e-7)
            inter = (pred * all_seg_masks).sum(dim=(1, 2))
            union = pred.sum(dim=(1, 2)) + all_seg_masks.sum(dim=(1, 2))
            d = ((2 * inter + 1e-6) / (union + 1e-6)).mean().item()
            thresh_dices.append(d)
            thresh_precisions.append(p)
            thresh_recalls.append(r)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, thresh_dices, 'g-o', label='Dice', markersize=5)
    ax.plot(thresholds, thresh_precisions, 'b-s', label='Precision', markersize=5)
    ax.plot(thresholds, thresh_recalls, 'r-^', label='Recall', markersize=5)
    best_thresh_idx = np.argmax(thresh_dices)
    ax.axvline(thresholds[best_thresh_idx], color='gray', linestyle='--', alpha=0.7,
               label=f'Best threshold = {thresholds[best_thresh_idx]:.2f}')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Segmenter: Metrics vs Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.05, 1.0])
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(CHART_DIR / "segmenter_threshold_analysis.png", dpi=150)
    plt.close()
    print(f"Saved: segmenter_threshold_analysis.png")


    # INFERENCE TIME BENCHMARKING
    print("\nRunning inference benchmarks...")

    # classifier benchmark
    dummy_cls = torch.randn(1, 3, 224, 224).to(device)
    # warmup
    for _ in range(10):
        with torch.no_grad():
            cls_model(dummy_cls)
    cls_times = []
    for _ in range(100):
        start = time.perf_counter()
        with torch.no_grad():
            cls_model(dummy_cls)
        cls_times.append((time.perf_counter() - start) * 1000)  # ms
    cls_time_mean = np.mean(cls_times)
    cls_time_std = np.std(cls_times)

    # segmenter benchmark
    dummy_seg = torch.randn(1, 1, 256, 256).to(device)
    for _ in range(10):
        with torch.no_grad():
            seg_model(dummy_seg)
    seg_times = []
    for _ in range(100):
        start = time.perf_counter()
        with torch.no_grad():
            seg_model(dummy_seg)
        seg_times.append((time.perf_counter() - start) * 1000)
    seg_time_mean = np.mean(seg_times)
    seg_time_std = np.std(seg_times)

    # ONNX benchmark
    onnx_cls_time_str = "N/A"
    onnx_seg_time_str = "N/A"
    try:
        import onnxruntime as ort
        cls_onnx_path = OUT_DIR / "stroke_type_classifier_single.onnx"
        seg_onnx_path = OUT_DIR / "stroke_location_segmenter_single.onnx"

        if cls_onnx_path.exists():
            sess_cls = ort.InferenceSession(str(cls_onnx_path))
            dummy_np = np.random.randn(1, 3, 224, 224).astype(np.float32)
            for _ in range(10):
                sess_cls.run(None, {"input": dummy_np})
            onnx_cls_times = []
            for _ in range(100):
                start = time.perf_counter()
                sess_cls.run(None, {"input": dummy_np})
                onnx_cls_times.append((time.perf_counter() - start) * 1000)
            onnx_cls_time_str = f"{np.mean(onnx_cls_times):.2f} +/- {np.std(onnx_cls_times):.2f} ms"

        if seg_onnx_path.exists():
            sess_seg = ort.InferenceSession(str(seg_onnx_path))
            dummy_np = np.random.randn(1, 1, 256, 256).astype(np.float32)
            for _ in range(10):
                sess_seg.run(None, {"input": dummy_np})
            onnx_seg_times = []
            for _ in range(100):
                start = time.perf_counter()
                sess_seg.run(None, {"input": dummy_np})
                onnx_seg_times.append((time.perf_counter() - start) * 1000)
            onnx_seg_time_str = f"{np.mean(onnx_seg_times):.2f} +/- {np.std(onnx_seg_times):.2f} ms"
    except ImportError:
        print("onnxruntime not installed — skipping ONNX benchmarks")


    # COMPUTE ALL CLASSIFIER METRICS
    # per-class specificity
    specificities = []
    for i in range(3):
        tn = cm[np.arange(3) != i][:, np.arange(3) != i].sum()
        fp = cm[:, i].sum() - cm[i, i]
        specificities.append(tn / (tn + fp + 1e-7))

    # weighted averages
    cls_precision_macro = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    cls_recall_macro = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    cls_f1_macro = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    cls_f1_weighted = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    kappa = cohen_kappa_score(all_targets, all_preds)

    # class distribution
    class_counts = np.bincount(all_targets, minlength=3)

    print(f"\nAll charts saved to: {CHART_DIR.resolve()}")

    # =========================================================================
    # FULL EVALUATION REPORT
    # =========================================================================
    report = f"""========================================
  STROKE DETECTION EVALUATION REPORT
========================================

DATASET
------------------------------
Classifier val set: {len(cls_val)} images
  Normal:       {class_counts[0]}
  Ischemic:     {class_counts[1]}
  Hemorrhagic:  {class_counts[2]}
Segmenter val set:  {len(seg_val)} pairs

CLASSIFIER (ResNet18, 3-class)
------------------------------
Overall Accuracy:   {(all_preds == all_targets).mean():.4f}
Cohen's Kappa:      {kappa:.4f}

Per-class Precision:
  Normal:       {cls_precision[0]:.4f}
  Ischemic:     {cls_precision[1]:.4f}
  Hemorrhagic:  {cls_precision[2]:.4f}

Per-class Recall (Sensitivity):
  Normal:       {cls_recall[0]:.4f} ({int(cm[0,0])}/{int(cm[0].sum())})
  Ischemic:     {cls_recall[1]:.4f} ({int(cm[1,1])}/{int(cm[1].sum())})
  Hemorrhagic:  {cls_recall[2]:.4f} ({int(cm[2,2])}/{int(cm[2].sum())})

Per-class Specificity:
  Normal:       {specificities[0]:.4f}
  Ischemic:     {specificities[1]:.4f}
  Hemorrhagic:  {specificities[2]:.4f}

Per-class F1 Score:
  Normal:       {cls_f1[0]:.4f}
  Ischemic:     {cls_f1[1]:.4f}
  Hemorrhagic:  {cls_f1[2]:.4f}

Per-class AUC (one-vs-rest):
  Normal:       {aucs[0]:.4f}
  Ischemic:     {aucs[1]:.4f}
  Hemorrhagic:  {aucs[2]:.4f}

Per-class Average Precision:
  Normal:       {avg_precisions[0]:.4f}
  Ischemic:     {avg_precisions[1]:.4f}
  Hemorrhagic:  {avg_precisions[2]:.4f}

Macro Averages:
  Precision:    {cls_precision_macro:.4f}
  Recall:       {cls_recall_macro:.4f}
  F1:           {cls_f1_macro:.4f}
  AUC:          {np.mean(aucs):.4f}

Weighted F1:    {cls_f1_weighted:.4f}

Confusion Matrix:
                Pred Normal  Pred Ischemic  Pred Hemorrhagic
  Normal        {cm[0,0]:>6}         {cm[0,1]:>6}            {cm[0,2]:>6}
  Ischemic      {cm[1,0]:>6}         {cm[1,1]:>6}            {cm[1,2]:>6}
  Hemorrhagic   {cm[2,0]:>6}         {cm[2,1]:>6}            {cm[2,2]:>6}

Clinical Safety:
  Strokes missed as Normal: {int(cm[1,0] + cm[2,0])} / {int(cm[1].sum() + cm[2].sum())}
  False stroke rate (Normal predicted as stroke): {int(cm[0,1] + cm[0,2])} / {int(cm[0].sum())}

SEGMENTER (UNet, binary mask)
------------------------------
Mean Dice:          {seg_dice_mean:.4f} +/- {seg_dice_std:.4f}
Mean IoU:           {seg_iou_mean:.4f} +/- {seg_iou_std:.4f}
Pixel Accuracy:     {seg_pixel_acc:.4f}
Pixel Precision:    {seg_precision:.4f}
Pixel Recall:       {seg_recall:.4f}
Pixel Specificity:  {seg_specificity:.4f}
Pixel F1:           {seg_f1:.4f}

Pixel Confusion:
  TP: {seg_TP:,.0f}   FP: {seg_FP:,.0f}
  FN: {seg_FN:,.0f}   TN: {seg_TN:,.0f}

Best Threshold (by Dice): {thresholds[best_thresh_idx]:.2f}
  Dice at best:     {thresh_dices[best_thresh_idx]:.4f}
  Precision:        {thresh_precisions[best_thresh_idx]:.4f}
  Recall:           {thresh_recalls[best_thresh_idx]:.4f}

INFERENCE PERFORMANCE
------------------------------
PyTorch (single image):
  Classifier:   {cls_time_mean:.2f} +/- {cls_time_std:.2f} ms
  Segmenter:    {seg_time_mean:.2f} +/- {seg_time_std:.2f} ms

ONNX Runtime (single image):
  Classifier:   {onnx_cls_time_str}
  Segmenter:    {onnx_seg_time_str}

Device: {device}

========================================
"""

    print(report)
    
    report_path = CHART_DIR / "evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved: evaluation_report.txt")

    # also save as JSON for programmatic access
    metrics_json = {
        "classifier": {
            "accuracy": float((all_preds == all_targets).mean()),
            "cohens_kappa": float(kappa),
            "per_class": {
                label: {
                    "precision": float(cls_precision[i]),
                    "recall": float(cls_recall[i]),
                    "specificity": float(specificities[i]),
                    "f1": float(cls_f1[i]),
                    "auc": float(aucs[i]),
                    "avg_precision": float(avg_precisions[i]),
                    "support": int(class_counts[i]),
                }
                for i, label in enumerate(labels)
            },
            "macro_precision": float(cls_precision_macro),
            "macro_recall": float(cls_recall_macro),
            "macro_f1": float(cls_f1_macro),
            "macro_auc": float(np.mean(aucs)),
            "weighted_f1": float(cls_f1_weighted),
            "strokes_missed_as_normal": int(cm[1, 0] + cm[2, 0]),
            "total_stroke_samples": int(cm[1].sum() + cm[2].sum()),
        },
        "segmenter": {
            "mean_dice": seg_dice_mean,
            "std_dice": seg_dice_std,
            "mean_iou": seg_iou_mean,
            "std_iou": seg_iou_std,
            "pixel_accuracy": float(seg_pixel_acc),
            "pixel_precision": float(seg_precision),
            "pixel_recall": float(seg_recall),
            "pixel_specificity": float(seg_specificity),
            "pixel_f1": float(seg_f1),
            "best_threshold": float(thresholds[best_thresh_idx]),
        },
        "inference_ms": {
            "pytorch_classifier": float(cls_time_mean),
            "pytorch_segmenter": float(seg_time_mean),
            "onnx_classifier": onnx_cls_time_str,
            "onnx_segmenter": onnx_seg_time_str,
        },
        "device": str(device),
    }

    with open(CHART_DIR / "evaluation_metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"Saved: evaluation_metrics.json")


if __name__ == "__main__":
    main()