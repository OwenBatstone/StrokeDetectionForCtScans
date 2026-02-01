import os
import zipfile
import random
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import models, transforms
from tqdm import tqdm

#Utilities / helper functions

def set_seed(seed: int = 42):
    """
    Set all random seeds so shuffling + initialization is repeatable.
    (Still not 100% identical across every GPU setup, but close enough.)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unzip_if_needed(zip_path: Path, out_dir: Path):
    """
    Extract a zip file into out_dir, but only if it hasn't already been extracted.
    I used zips because it's just easier to move datasets around that way.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    #If we can already find PNGs in here, assume it's extracted and move on.
    if any(out_dir.rglob("*.png")):
        return

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)


def pil_open_rgb(p: Path) -> Image.Image:
    return Image.open(p).convert("RGB")


def pil_open_gray(p: Path) -> Image.Image:
    return Image.open(p).convert("L")


def build_binary_mask_from_overlay(original_rgb: Image.Image, overlay_rgb: Image.Image, diff_thresh: int = 25):
    """
    Create a binary mask (0/1) by comparing the original CT slice to its overlay version.

    Idea:
    - The overlay image has the stroke region highlighted in color.
    - If a pixel changed a lot compared to the original, it's probably part of the stroke highlight.
    """
    orig = np.array(original_rgb, dtype=np.int16)   #original slice
    over = np.array(overlay_rgb, dtype=np.int16)    #overlay slice (with colored highlight)

    # Sum absolute RGB differences per pixel: bigger = more "changed"
    diff = np.abs(over - orig).sum(axis=2)

    # Anything above threshold becomes stroke = 1
    mask = (diff > diff_thresh).astype(np.uint8)   
    return mask


# Datasets

class StrokeClassificationDataset(Dataset):
    """
    3-class image classification dataset:
      0 = Normal
      1 = Ischemic
      2 = Hemorrhagic
    """
    def __init__(self, normal_dir: Path, ischemic_dir: Path, hemorr_dir: Path, transform=None):
        self.transform = transform
        self.samples = []

        # Build a list of path, label
        for p in sorted(normal_dir.glob("*.png")):
            self.samples.append((p, 0))
        for p in sorted(ischemic_dir.glob("*.png")):
            self.samples.append((p, 1))
        for p in sorted(hemorr_dir.glob("*.png")):
            self.samples.append((p, 2))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        #Grab the path and label
        p, y = self.samples[idx]

        #ResNet expects 3 channels, RGB
        img = pil_open_rgb(p)

        #Apply transforms 
        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(y, dtype=torch.long)


class StrokeSegmentationDataset(Dataset):
    """
    Segmentation dataset: learn where the stroke is (pixel-wise).
    Inputs:  original CT slice (we feed grayscale for simplicity)
    Targets: a binary mask derived from overlay-vs-original differences
    Note: We only use ischemic + hemorrhagic for segmentation, because normal has no stroke highlight.
    """
    def __init__(
        self,
        ischemic_dir: Path,
        hemorr_dir: Path,
        ischemic_overlay_dir: Path,
        hemorr_overlay_dir: Path,
        img_transform=None,
        mask_size=(256, 256),
        diff_thresh=25
    ):
        self.img_transform = img_transform
        self.mask_size = mask_size
        self.diff_thresh = diff_thresh

        self.pairs = []

        #Pair each original image with its matching overlay by filename
        for img_dir, ov_dir in [(ischemic_dir, ischemic_overlay_dir), (hemorr_dir, hemorr_overlay_dir)]:
            for img_path in sorted(img_dir.glob("*.png")):
                ov_path = ov_dir / img_path.name
                if ov_path.exists():
                    self.pairs.append((img_path, ov_path))

        if len(self.pairs) == 0:
            raise RuntimeError("No segmentation pairs found (check overlay filenames match originals).")

        # Not strictly needed might remove
        self._mask_resize = transforms.Resize(
            self.mask_size,
            interpolation=transforms.InterpolationMode.NEAREST
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, ov_path = self.pairs[idx]

        #Load both as RGB so we can compute pixel differences
        orig_rgb = pil_open_rgb(img_path)
        ov_rgb = pil_open_rgb(ov_path)

        #Build binary mask 
        mask = build_binary_mask_from_overlay(orig_rgb, ov_rgb, diff_thresh=self.diff_thresh)

        #UNet input: use grayscale CT slice
        orig_gray = pil_open_gray(img_path)

        #Resize image to training resolution
        orig_gray = orig_gray.resize(self.mask_size[::-1], resample=Image.BILINEAR)

        #Resize the mask to
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
        mask_pil = mask_pil.resize(self.mask_size[::-1], resample=Image.NEAREST)

        #Convert to tensors
        img_t = transforms.ToTensor()(orig_gray)   
        mask_t = transforms.ToTensor()(mask_pil)   

        #Force the mask to be clean 0/1 and drop the channel dimension
        mask_t = (mask_t > 0.5).float().squeeze(0)

        #Optional normalization
        if self.img_transform:
            img_t = self.img_transform(img_t)

        return img_t, mask_t


def split_indices(n, val_frac=0.15, seed=42):
    #helper to split dataset indices into train/val.
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)

    val_n = int(round(n * val_frac))

    # train indices first, then val indices
    return idx[val_n:], idx[:val_n]


class SubsetDataset(Dataset):
    def __init__(self, base_ds, indices):
        self.base_ds = base_ds
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.base_ds[self.indices[i]]


#Models

def make_resnet18_classifier(num_classes=3, pretrained=True):
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    in_features = m.fc.in_features
    m.fc = nn.Linear(in_features, num_classes)
    return m


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetSmall(nn.Module):
    def __init__(self, in_ch=1, base=32):
        super().__init__()

        # Down / encoder path
        self.down1 = DoubleConv(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)

        # ottleneck
        self.mid = DoubleConv(base * 4, base * 8)

        #decoder path
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.conv3 = DoubleConv(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.conv2 = DoubleConv(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.conv1 = DoubleConv(base * 2, base)

        #stroke logits
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        #Encoder
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))

        #Bottleneck
        m = self.mid(self.pool3(d3))

        #Decoder
        u3 = self.up3(m)
        u3 = torch.cat([u3, d3], dim=1)
        u3 = self.conv3(u3)

        u2 = self.up2(u3)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.conv2(u2)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.conv1(u1)

        return self.out(u1)


#Training

def train_classifier(model, train_loader, val_loader, device, epochs=5, lr=1e-4):
    """
    Train the ResNet classifier (Normal / Ischemic / Hemorrhagic).
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    best_val = 0.0

    for ep in range(1, epochs + 1):
        model.train()
        total, correct = 0, 0
        loss_sum = 0.0

        for x, y in tqdm(train_loader, desc=f"[CLS] Epoch {ep}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)

            #Mixed precision on GPU
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(x)
                loss = F.cross_entropy(logits, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            loss_sum += loss.item() * x.size(0)
            total += x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()

        train_acc = correct / max(1, total)

        # Validation pass
        model.eval()
        vtotal, vcorrect = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                vtotal += x.size(0)
                vcorrect += (logits.argmax(dim=1) == y).sum().item()

        val_acc = vcorrect / max(1, vtotal)
        print(f"[CLS] Epoch {ep}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

        best_val = max(best_val, val_acc)

    print(f"[CLS] Done. Best val_acc ~ {best_val:.4f}")


def dice_loss_from_logits(logits, targets, eps=1e-6):
    """
    Dice loss is great for segmentation because stroke pixels are usually a small fraction of the image.

    logits:  [B,1,H,W]
    targets: [B,H,W] in {0,1}
    """
    probs = torch.sigmoid(logits).squeeze(1)  # -> [B,H,W]
    targets = targets.float()

    inter = (probs * targets).sum(dim=(1, 2))
    union = probs.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))

    dice = (2 * inter + eps) / (union + eps)
    return 1 - dice.mean()


def train_segmenter(model, train_loader, val_loader, device, epochs=5, lr=1e-3):
    """
    Train the UNet segmenter.
    Loss = 50% BCE-with-logits + 50% Dice loss (nice balance for binary masks).
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    best_val = 1e9

    for ep in range(1, epochs + 1):
        model.train()
        loss_sum, n = 0.0, 0

        for x, mask in tqdm(train_loader, desc=f"[SEG] Epoch {ep}/{epochs}", leave=False):
            x, mask = x.to(device), mask.to(device)
            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(x)  # [B,1,H,W]

                #Pixel-wise loss + overlap-based loss
                bce = F.binary_cross_entropy_with_logits(logits.squeeze(1), mask)
                dsc = dice_loss_from_logits(logits, mask)
                loss = 0.5 * bce + 0.5 * dsc

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            loss_sum += loss.item() * x.size(0)
            n += x.size(0)

        train_loss = loss_sum / max(1, n)

        # Validation pass
        model.eval()
        vloss_sum, vn = 0.0, 0
        with torch.no_grad():
            for x, mask in val_loader:
                x, mask = x.to(device), mask.to(device)
                logits = model(x)

                bce = F.binary_cross_entropy_with_logits(logits.squeeze(1), mask)
                dsc = dice_loss_from_logits(logits, mask)
                loss = 0.5 * bce + 0.5 * dsc

                vloss_sum += loss.item() * x.size(0)
                vn += x.size(0)

        val_loss = vloss_sum / max(1, vn)
        print(f"[SEG] Epoch {ep}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        best_val = min(best_val, val_loss)

    print(f"[SEG] Done. Best val_loss ~ {best_val:.4f}")

#ONNX export

def export_classifier_onnx(model, out_path: Path, device):
    """
    Export classifier to ONNX so Flutter (onnxruntime) can run it.
    """
    model.eval().to(device)

    #Dummy input that matches the classifier input shape
    dummy = torch.randn(1, 3, 224, 224, device=device)

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )

    print(f"[ONNX] Saved classifier -> {out_path}")


def export_segmenter_onnx(model, out_path: Path, device, h=256, w=256):
    """
    Export segmenter to ONNX.
    Output is mask logits (you sigmoid + threshold later in Flutter).
    """
    model.eval().to(device)

    #Dummy input matching UNet shape
    dummy = torch.randn(1, 1, h, w, device=device)

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["mask_logits"],
        dynamic_axes={"input": {0: "batch"}, "mask_logits": {0: "batch"}},
    )

    print(f"[ONNX] Saved segmenter -> {out_path}")


def main():
    """
    End-to-end pipeline:
      1) unzip datasets
      2) find the real PNG folders
      3) train classifier (ResNet18) -> export ONNX
      4) train segmenter (UNet) -> export ONNX
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="extracted_data",
                        help="Where to extract the zips")
    parser.add_argument("--out_dir", type=str, default="onnx_out",
                        help="Where to save ONNX models")

    parser.add_argument("--epochs_cls", type=int, default=5)
    parser.add_argument("--epochs_seg", type=int, default=5)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--no_pretrained", action="store_true",
                        help="If set, do NOT use ImageNet pretrained weights for ResNet18")

    parser.add_argument("--diff_thresh", type=int, default=25,
                        help="How different overlay-vs-original must be to call it 'stroke'")

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dataset zip files (expected in the same folder as this script)
    z_hem = Path("Hemorrhagic Stroke.zip")
    z_isc = Path("Ischemic Stroke.zip")
    z_nor = Path("Normal.zip")
    z_isc_ov = Path("Ischemic Overlay.zip")
    z_hem_ov = Path("Hemorrhagic Overlay.zip")

    #if any zip is missing, fail early with a clear message
    for z in [z_hem, z_isc, z_nor, z_isc_ov, z_hem_ov]:
        if not z.exists():
            raise FileNotFoundError(
                f"Missing {z}. Put the zip in the same folder as this script, "
                f"or edit the zip paths in the script."
            )

    # Extract zips (only if needed)
    unzip_if_needed(z_hem, data_root / "Hemorrhagic Stroke")
    unzip_if_needed(z_isc, data_root / "Ischemic Stroke")
    unzip_if_needed(z_nor, data_root / "Normal")
    unzip_if_needed(z_isc_ov, data_root / "Ischemic overlay")
    unzip_if_needed(z_hem_ov, data_root / "Hemorrhagic Overlay")

    def find_png_dir(root: Path) -> Path:
        """
        Some zips include an extra nested folder layer.
        This finds the folder that actually contains the PNGs (and picks the biggest one).
        """
        png_dirs = sorted({p.parent for p in root.rglob("*.png")})
        if not png_dirs:
            raise RuntimeError(f"No PNG files found under {root}")
        best = max(png_dirs, key=lambda d: len(list(d.glob("*.png"))))
        return best

    # Find the real directories that contain images
    hemorr_dir = find_png_dir(data_root / "Hemorrhagic Stroke")
    ischemic_dir = find_png_dir(data_root / "Ischemic Stroke")
    normal_dir = find_png_dir(data_root / "Normal")
    ischemic_overlay_dir = find_png_dir(data_root / "Ischemic overlay")
    hemorr_overlay_dir = find_png_dir(data_root / "Hemorrhagic Overlay")

    print("Found dirs:")
    print("  Normal:", normal_dir)
    print("  Ischemic:", ischemic_dir)
    print("  Hemorrhagic:", hemorr_dir)
    print("  Ischemic Overlay:", ischemic_overlay_dir)
    print("  Hemorrhagic Overlay:", hemorr_overlay_dir)

    #Classification: ResNet18

    # Preprocessing for classifier (224x224 + ImageNet normalization)
    cls_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    cls_ds = StrokeClassificationDataset(normal_dir, ischemic_dir, hemorr_dir, transform=cls_tf)
    tr_idx, va_idx = split_indices(len(cls_ds), val_frac=0.15, seed=args.seed)

    cls_train = SubsetDataset(cls_ds, tr_idx)
    cls_val = SubsetDataset(cls_ds, va_idx)

    cls_train_loader = DataLoader(cls_train, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)
    cls_val_loader = DataLoader(cls_val, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    # Build + train the classifier
    cls_model = make_resnet18_classifier(num_classes=3, pretrained=(not args.no_pretrained)).to(device)
    train_classifier(cls_model, cls_train_loader, cls_val_loader, device, epochs=args.epochs_cls, lr=1e-4)

    # Export classifier to ONNX for Flutter
    export_classifier_onnx(cls_model, out_dir / "stroke_type_classifier.onnx", device)

    #Segmentation: UNet

    #Simple normalization for 1-channel CT slice inputs
    seg_img_tf = transforms.Normalize(mean=[0.5], std=[0.5])

    seg_ds = StrokeSegmentationDataset(
        ischemic_dir=ischemic_dir,
        hemorr_dir=hemorr_dir,
        ischemic_overlay_dir=ischemic_overlay_dir,
        hemorr_overlay_dir=hemorr_overlay_dir,
        img_transform=seg_img_tf,
        mask_size=(256, 256),
        diff_thresh=args.diff_thresh
    )

    tr_idx, va_idx = split_indices(len(seg_ds), val_frac=0.15, seed=args.seed)
    seg_train = SubsetDataset(seg_ds, tr_idx)
    seg_val = SubsetDataset(seg_ds, va_idx)

    # Segmentation is heavier than classification, so use a smaller batch size
    seg_bs = max(4, args.batch // 4)

    seg_train_loader = DataLoader(seg_train, batch_size=seg_bs, shuffle=True, num_workers=2, pin_memory=True)
    seg_val_loader = DataLoader(seg_val, batch_size=seg_bs, shuffle=False, num_workers=2, pin_memory=True)

    seg_model = UNetSmall(in_ch=1, base=32).to(device)
    train_segmenter(seg_model, seg_train_loader, seg_val_loader, device, epochs=args.epochs_seg, lr=1e-3)

    # Export segmenter to ONNX for Flutter
    export_segmenter_onnx(seg_model, out_dir / "stroke_location_segmenter.onnx", device, h=256, w=256)

    print("\nAll done.")
    print("ONNX models saved to:", out_dir.resolve())
    print("Files:")
    print(" - stroke_type_classifier.onnx")
    print(" - stroke_location_segmenter.onnx")


if __name__ == "__main__":
    main()
