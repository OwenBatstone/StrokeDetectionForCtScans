#imports
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
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pydicom as dicom
import cv2



#Utilities

def set_seed(seed: int = 42): #sets all rng seeds to 42 for repeatability
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def dicom_to_images(input_folder, output_folder, use_clahe=True):
    os.makedirs(output_folder, exist_ok=True)

    def apply_window(img, center, width):
        lower = center - width / 2
        upper = center + width / 2
        img = np.clip(img, lower, upper)
        img = (img - lower) / (upper - lower) * 255.0
        return img.astype(np.uint8)

    for n, filename in enumerate(os.listdir(input_folder)):
        if not filename.lower().endswith(".dcm"):
            continue

        try:
            ds = dicom.dcmread(os.path.join(input_folder, filename))
            img = ds.pixel_array.astype(np.float32)

            # Convert to Hounsfield Units
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                img = img * float(ds.RescaleSlope) + float(ds.RescaleIntercept)

            # Use DICOM window if available
            if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
                wc = ds.WindowCenter
                ww = ds.WindowWidth
                wc = wc[0] if isinstance(wc, dicom.multival.MultiValue) else wc
                ww = ww[0] if isinstance(ww, dicom.multival.MultiValue) else ww
                img = apply_window(img, float(wc), float(ww))
            else:
                # Brain hemorrhage fallback window
                img = apply_window(img, center=60, width=150)

            if use_clahe:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                img = clahe.apply(img)

            out_path = os.path.join(output_folder, filename.replace(".dcm", ".png"))
            cv2.imwrite(out_path, img)

            if n % 50 == 0:
                print(f"{n} images converted")

        except Exception as e:
            print(f"Skipped {filename}: {e}")


dicom_to_images(
    input_folder=r"C:\DeepLearningTest\Brain_Stroke_CT_Dataset\Bleeding\DICOM",
    output_folder=r"C:\DeepLearningTest\Data\ConvertedDICOM"
)




def unzip_if_needed(zip_path: Path, out_dir: Path): #if a zip file is given unzips it
    out_dir.mkdir(parents=True, exist_ok=True) #creates output dir if it doesnt exist
    if any(out_dir.rglob("*.png")):
        return #if theres a png then we assume its already extracted
    with zipfile.ZipFile(zip_path, "r") as z: #open zip file for reading
        z.extractall(out_dir) #puts all the file in the output dir


def pil_open_rgb(p: Path) -> Image.Image: #opens the image as RGB
    #Strips alpha cleanly
    return Image.open(p).convert("RGB")


def pil_open_gray(p: Path) -> Image.Image: #opens image as a greyschgale
    return Image.open(p).convert("L")


def find_png_dir(root: Path) -> Path: #fines the best directory containing png files
    png_dirs = sorted({p.parent for p in root.rglob("*.png")}) #gets all directories with pngs
    if not png_dirs: #error if none found
        raise RuntimeError(f"No PNG files found under {root}")
    best = max(png_dirs, key=lambda d: len(list(d.glob("*.png")))) #takes directory with the most pngs
    return best


def build_binary_mask_from_overlay(original_rgb: Image.Image, overlay_rgb: Image.Image, diff_thresh: int = 25, cleanup: bool = True):
    orig = np.asarray(original_rgb, dtype=np.int16) #converts image to numpy array
    over = np.asarray(overlay_rgb, dtype=np.int16) #converts overlay to numpy array

    diff = np.abs(over - orig).sum(axis=2)  #per pixel RBG difference
    mask = (diff > diff_thresh).astype(np.uint8) #if difference meets threshold, 1, if not 0

    if not cleanup: #if the cleanup is disabled then we return the mask immediatly
        return mask

    #Convert to tensor [1,1,H,W]
    t = torch.from_numpy(mask[None, None, :, :].astype(np.float32))

    # Dilation then erosion to fill holes
    t = F.max_pool2d(t, kernel_size=3, stride=1, padding=1)
    #invert, dilate, invert
    t = 1.0 - F.max_pool2d(1.0 - t, kernel_size=3, stride=1, padding=1)

    #Remove isolated noise by another mild opening
    t = 1.0 - F.max_pool2d(1.0 - t, kernel_size=3, stride=1, padding=1)
    t = F.max_pool2d(t, kernel_size=3, stride=1, padding=1)

    out = (t.squeeze().numpy() > 0.5).astype(np.uint8)
    return out


#Datasets

class StrokeClassificationDataset(Dataset): #COME HERE TO MAKE PREDICTION MODEL BETTER :D
    #0 = Normal, 1 = ischemic, 2 = hemorrhagic
    def __init__(self, normal_dir: Path, ischemic_dir: Path, hemorr_dir: Path, transform=None):
        self.transform = transform #may be nothing, if a resize+totensor is passed
        self.samples = []

        #iterating throuhg the different png paths, then labels them for future predictions
        for p in sorted(normal_dir.glob("*.png")):
            self.samples.append((p, 0))
        for p in sorted(ischemic_dir.glob("*.png")):
            self.samples.append((p, 1))
        for p in sorted(hemorr_dir.glob("*.png")):
            self.samples.append((p, 2))

        if not self.samples:
            raise RuntimeError("No classification images found.")

    def __len__(self): #gets dataset length
        return len(self.samples)

    def __getitem__(self, idx): #get one sample
        p, y = self.samples[idx]
        im = pil_open_rgb(p)
        if self.transform:
            im = self.transform(im)
        return im, torch.tensor(y, dtype=torch.long)


class StrokeSegmentationDataset(Dataset): #dataset for binary segmanetation, only for the stroke types, normal will obviously not have an overlay (if you think it would you may be dumb sorry ;( )
    """
    Binary segmentation for stroke region detection later on
    Uses the original and overlay and creates a binary mask by the difference in pixels.
    """
    def __init__( #constructor with the original and overlays 
        self,
        ischemic_dir: Path,
        hemorr_dir: Path,
        ischemic_overlay_dir: Path,
        hemorr_overlay_dir: Path,
        size=(256, 256), #resizes the taarget 
        diff_thresh=25, #threshold to mask difference
        cleanup=True, #runs morphological cleanup
    ):
        self.size = size #sotres the ressized size
        self.diff_thresh = diff_thresh #stores the difference threshold
        self.cleanup = cleanup #stores cleanup flag

        self.pairs = [] #holds the pairs of overlay x original (love story)
        for img_dir, ov_dir in [(ischemic_dir, ischemic_overlay_dir), (hemorr_dir, hemorr_overlay_dir)]:
            for img_path in sorted(img_dir.glob("*.png")): #llops through og images
                ov_path = ov_dir / img_path.name #overlays by maching the two naems
                if ov_path.exists(): #only keep pairs where the overlay exists for the image
                    self.pairs.append((img_path, ov_path))

        if not self.pairs: #if no overlays exist
            raise RuntimeError("No pairs found. Name your files better or like get overlays... This is like the entire point, get a better dataset or smt idk")

        self.to_tensor = transforms.ToTensor() #converts to tensor

    def __len__(self): #required length
        return len(self.pairs) #number of pairs

    def __getitem__(self, idx): #gets one training example
        img_path, ov_path = self.pairs[idx]

        #loads images as RGV
        orig_rgb = pil_open_rgb(img_path) 
        ov_rgb = pil_open_rgb(ov_path)

        mask = build_binary_mask_from_overlay( #builds masks based off the overlay differences
            orig_rgb, ov_rgb,
            diff_thresh=self.diff_thresh,
            cleanup=self.cleanup
        )  #uint8 {0,1} [H,W]

        #grayscale input
        orig_gray = pil_open_gray(img_path)

        #resize both
        W, H = self.size[1], self.size[0]  # PIL uses (W,H)
        orig_gray = orig_gray.resize((W, H), resample=Image.BILINEAR)

        mask_pil = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
        mask_pil = mask_pil.resize((W, H), resample=Image.NEAREST)

        #tensors
        img_t = self.to_tensor(orig_gray)           #[1,H,W] float 0..1
        mask_t = self.to_tensor(mask_pil).squeeze(0)  #[H,W] float 0..1
        mask_t = (mask_t > 0.5).float()             #hard {0,1}

        return img_t, mask_t


def split_indices(n, val_frac=0.15, seed=42): #splits into train/validation lists
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    val_n = int(round(n * val_frac))
    return idx[val_n:], idx[:val_n]


class SubsetDataset(Dataset): #wrapper to view a subset of anmother dataset
    def __init__(self, base_ds, indices):
        self.base_ds = base_ds
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.base_ds[self.indices[i]] #maps subset index to base dataset index


#Models

def make_resnet18_classifier(num_classes=3, pretrained=True): #ResNet18 classifier head
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


class DoubleConv(nn.Module): #Unet block
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

    def forward(self, x): #passes it forwards
        return self.net(x)


class UNetSmall(nn.Module): #small UNet for 256x256 grayscall segmentation
    def __init__(self, in_ch=1, base=32):
        super().__init__()
        self.down1 = DoubleConv(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.mid = DoubleConv(base * 4, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.conv3 = DoubleConv(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.conv2 = DoubleConv(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.conv1 = DoubleConv(base * 2, base)

        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x): #forward pass through Unet
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))

        m = self.mid(self.pool3(d3))

        u3 = self.up3(m)
        u3 = torch.cat([u3, d3], dim=1)
        u3 = self.conv3(u3)

        u2 = self.up2(u3)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.conv2(u2)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.conv1(u1)

        return self.out(u1)  # [B,1,H,W] logits


#Training

def train_classifier(model, train_loader, val_loader, device, epochs=5, lr=1e-4): #training classifier loop
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    best_val = 0.0

    for ep in range(1, epochs + 1): #loops over epoches (starts at 1)
        model.train()
        total, correct = 0, 0
        loss_sum = 0.0

        for x, y in tqdm(train_loader, desc=f"[CLS] Epoch {ep}/{epochs}", leave=False): #batch loop with progress bar :D
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16): #lets you use cuda if possible (sucks to be a amd gpu user rn... takes me 2 hours ;-;)
                logits = model(x)
                loss = F.cross_entropy(logits, y)

            scaler.scale(loss).backward() #prevents underflow
            scaler.step(opt)
            scaler.update()

            loss_sum += loss.item() * x.size(0) #accumulates total loss over the samples
            total += x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()

        train_acc = correct / max(1, total)

        model.eval() #sets the model to eval mode
        vtotal, vcorrect = 0, 0 #validation counters
        with torch.no_grad():
            for x, y in val_loader: #loop through validation data
                x, y = x.to(device), y.to(device)
                logits = model(x)
                vtotal += x.size(0)
                vcorrect += (logits.argmax(dim=1) == y).sum().item()

        val_acc = vcorrect / max(1, vtotal) #validation accuracy
        print(f"[CLS] Epoch {ep}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}") #reports epoch stats
        best_val = max(best_val, val_acc) #update best validation accuracy

    print(f"[CLS] Done :D. Best val_acc: {best_val:.4f}") #final training summary

def evaluate_classifier(model, loader, device):
    """
    Runs full evaluation and prints:
    - confusion matrix
    - precision / recall / f1
    - overall accuracy
    """

    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)

            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(y.numpy())

    #metrics
    acc = accuracy_score(all_targets, all_preds)
    cm = confusion_matrix(all_targets, all_preds)

    print("\nCLASSIFICATION REPORT:")
    print(f"Overall Accuracy: {acc:.4f}\n")

    print("Confusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(
        all_targets,
        all_preds,
        target_names=["Normal", "Ischemic", "Hemorrhagic"]
    ))

def dice_loss_from_logits(logits, targets, eps=1e-6): #dice loss for segmentation
    probs = torch.sigmoid(logits).squeeze(1)  # [B,H,W]
    targets = targets.float()

    inter = (probs * targets).sum(dim=(1, 2))
    union = probs.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))
    dice = (2 * inter + eps) / (union + eps)
    return 1 - dice.mean()


def train_segmenter(model, train_loader, val_loader, device, epochs=5, lr=1e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    best_val = 1e9

    for ep in range(1, epochs + 1):
        model.train() #train mode
        loss_sum, n = 0.0, 0 #accumlate lsoss nad sample count

        for x, mask in tqdm(train_loader, desc=f"[SEG] Epoch {ep}/{epochs}", leave=False): #train batches
            x, mask = x.to(device), mask.to(device) #mopve to device
            opt.zero_grad(set_to_none=True) #clear gradents

            with torch.amp.autocast(enabled=(device.type == "cuda")): #mixed precision on GPU
                logits = model(x)  # [B,1,H,W]
                bce = F.binary_cross_entropy_with_logits(logits.squeeze(1), mask)
                dsc = dice_loss_from_logits(logits, mask)
                loss = 0.5 * bce + 0.5 * dsc

            scaler.scale(loss).backward() #backprop scalled loss
            scaler.step(opt) #step optimizer
            scaler.update() #update scaler

            loss_sum += loss.item() * x.size(0) #accumulated weigteed loss
            n += x.size(0) #accumulated sample count

        train_loss = loss_sum / max(1, n) #average training loss

        model.eval() #evaluate mode
        vloss_sum, vn = 0.0, 0
        with torch.no_grad(): #no gradients
            for x, mask in val_loader: #validation batches
                x, mask = x.to(device), mask.to(device)
                logits = model(x)
                bce = F.binary_cross_entropy_with_logits(logits.squeeze(1), mask)
                dsc = dice_loss_from_logits(logits, mask)
                loss = 0.5 * bce + 0.5 * dsc
                vloss_sum += loss.item() * x.size(0)
                vn += x.size(0)

        val_loss = vloss_sum / max(1, vn) #averagve validation loss
        print(f"[SEG] Epoch {ep}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        best_val = min(best_val, val_loss)

    print(f"[SEG] Done :D. Best validation loss : {best_val:.4f}")

def evaluate_segmenter(model, loader, device, threshold=0.5, eps=1e-7):
    """
    Prints segmentation metrics on the validation set:
    - Dice, IoU
    - pixel precision/recall/F1
    - pixel confusion matrix (TP, FP, FN, TN)

    threshold: probability threshold after sigmoid
    """
    model.eval()

    # Accumulate pixel confusion totals across the whole val set
    TP = FP = FN = TN = 0

    dice_scores = []
    iou_scores = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)  # [B,H,W] in {0,1}

            logits = model(x)                 # [B,1,H,W]
            probs = torch.sigmoid(logits)     # [B,1,H,W]
            pred = (probs.squeeze(1) > threshold).float()  # [B,H,W]

            # ----- per-batch confusion -----
            tp = (pred * y).sum().item()
            fp = (pred * (1 - y)).sum().item()
            fn = ((1 - pred) * y).sum().item()
            tn = ((1 - pred) * (1 - y)).sum().item()

            TP += tp
            FP += fp
            FN += fn
            TN += tn

            # ----- Dice + IoU (batch-averaged) -----
            inter = (pred * y).sum(dim=(1, 2))
            union = pred.sum(dim=(1, 2)) + y.sum(dim=(1, 2))
            dice = (2 * inter + eps) / (union + eps)

            iou = (inter + eps) / (pred.sum(dim=(1, 2)) + y.sum(dim=(1, 2)) - inter + eps)

            dice_scores.extend(dice.detach().cpu().tolist())
            iou_scores.extend(iou.detach().cpu().tolist())

    # Dataset-level metrics from accumulated confusion
    precision = TP / (TP + FP + eps)
    recall    = TP / (TP + FN + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    acc       = (TP + TN) / (TP + TN + FP + FN + eps)

    dice_mean = float(np.mean(dice_scores)) if dice_scores else 0.0
    iou_mean  = float(np.mean(iou_scores)) if iou_scores else 0.0

    print("\nSEGMENTATION REPORT:")
    print(f"Threshold: {threshold}")
    print(f"Pixel Accuracy: {acc:.4f}")
    print(f"Pixel Precision: {precision:.4f}")
    print(f"Pixel Recall:    {recall:.4f}")
    print(f"Pixel F1:        {f1:.4f}")
    print(f"Mean Dice:       {dice_mean:.4f}")
    print(f"Mean IoU:        {iou_mean:.4f}")

    print("\nPixel Confusion (Totals over val set):")
    print(f"TP={TP:.0f}, FP={FP:.0f}, FN={FN:.0f}, TN={TN:.0f}")


#ONNX export for flutter (NO MORE .DATA PLEZZZZ)

def export_onnx_single_file(model, dummy, out_path: Path): #if this doesnt work i might rip my hair out, ive retrained this like 7 times and each have taken 2 hours...
    import onnx
    from onnx import external_data_helper

    out_path = Path(out_path) #ensures its a path
    tmp = out_path.with_suffix(".tmp.onnx") #temporary onnx path
    data_sidecar = Path(str(tmp) + ".data") #temporary sidecar path... but i need it to not stay or ill lose my marbles

    #clean old leftovers
    out_path.unlink(missing_ok=True)
    tmp.unlink(missing_ok=True)
    data_sidecar.unlink(missing_ok=True)
    Path(str(out_path) + ".data").unlink(missing_ok=True)

    #Export
    torch.onnx.export(
        model,
        dummy,
        str(tmp),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )

    #Load external data then save single-file
    m = onnx.load_model(str(tmp), load_external_data=True)
    external_data_helper.convert_model_from_external_data(m)
    onnx.save_model(m, str(out_path), save_as_external_data=False)

    #Cleanup temp
    tmp.unlink(missing_ok=True)
    data_sidecar.unlink(missing_ok=True)

    #No more sidecar, if i see a data file i might just have to drop out... its like 6 am, and ive been trying to get this to work since 11pm, im going to lose any sense of self i have left if thsi dosent work
    Path(str(out_path) + ".data").unlink(missing_ok=True)

    print(f"Single-file saved -> {out_path}")





#Main

def main(): #main function (where the magic happens ;) )
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="extracted_data")
    parser.add_argument("--out_dir", type=str, default="onnx_out")
    parser.add_argument("--epochs_cls", type=int, default=5)
    parser.add_argument("--epochs_seg", type=int, default=5)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--diff_thresh", type=int, default=25)
    parser.add_argument("--no_cleanup", action="store_true", help="Disable mask cleanup (speckle removal)")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    #zip file names **IMP, IF YOU ADD NEW ONES MAKE SURE TO COME HERE!!!!***
    z_hem = Path("Hemorrhagic Stroke.zip")
    z_isc = Path("Ischemic Stroke.zip")
    z_nor = Path("Normal.zip")
    z_isc_ov = Path("Ischemic Overlay.zip")
    z_hem_ov = Path("Hemorrhagic Overlay.zip")

    for z in [z_hem, z_isc, z_nor, z_isc_ov, z_hem_ov]:
        if not z.exists():
            raise FileNotFoundError(f"Missing {z}.")

    unzip_if_needed(z_hem, data_root / "Hemorrhagic Stroke")
    unzip_if_needed(z_isc, data_root / "Ischemic Stroke")
    unzip_if_needed(z_nor, data_root / "Normal")
    unzip_if_needed(z_isc_ov, data_root / "Ischemic Overlay")
    unzip_if_needed(z_hem_ov, data_root / "Hemorrhagic Overlay")

    hemorr_dir = find_png_dir(data_root / "Hemorrhagic Stroke")
    ischemic_dir = find_png_dir(data_root / "Ischemic Stroke")
    normal_dir = find_png_dir(data_root / "Normal")
    ischemic_overlay_dir = find_png_dir(data_root / "Ischemic Overlay")
    hemorr_overlay_dir = find_png_dir(data_root / "Hemorrhagic Overlay")

    print("Found dirs:")
    print("  Normal:", normal_dir)
    print("  Ischemic:", ischemic_dir)
    print("  Hemorrhagic:", hemorr_dir)
    print("  Ischemic Overlay:", ischemic_overlay_dir)
    print("  Hemorrhagic Overlay:", hemorr_overlay_dir)

    #CLASSIFIER transforms
    cls_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    cls_ds = StrokeClassificationDataset(normal_dir, ischemic_dir, hemorr_dir, transform=cls_tf)
    tr_idx, va_idx = split_indices(len(cls_ds), val_frac=0.15, seed=args.seed)
    cls_train = SubsetDataset(cls_ds, tr_idx)
    cls_val = SubsetDataset(cls_ds, va_idx)

    cls_train_loader = DataLoader(cls_train, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)
    cls_val_loader = DataLoader(cls_val, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)
    cls_model = make_resnet18_classifier(num_classes=3, pretrained=(not args.no_pretrained)).to(device)
    train_classifier(cls_model, cls_train_loader, cls_val_loader, device, epochs=args.epochs_cls, lr=1e-4)
    evaluate_classifier(cls_model, cls_val_loader, device)
    cls_model.eval()
    dummy_cls = torch.randn(1, 3, 224, 224, device=device)
    export_onnx_single_file(cls_model, dummy_cls, out_dir / "stroke_type_classifier_single.onnx")


    #SEGMENTER dataset
    seg_ds = StrokeSegmentationDataset(
        ischemic_dir=ischemic_dir,
        hemorr_dir=hemorr_dir,
        ischemic_overlay_dir=ischemic_overlay_dir,
        hemorr_overlay_dir=hemorr_overlay_dir,
        size=(256, 256),
        diff_thresh=args.diff_thresh,
        cleanup=(not args.no_cleanup),
    )

    tr_idx, va_idx = split_indices(len(seg_ds), val_frac=0.15, seed=args.seed)
    seg_train = SubsetDataset(seg_ds, tr_idx)
    seg_val = SubsetDataset(seg_ds, va_idx)

    seg_bs = max(4, args.batch // 4)
    seg_train_loader = DataLoader(seg_train, batch_size=seg_bs, shuffle=True, num_workers=2, pin_memory=True)
    seg_val_loader = DataLoader(seg_val, batch_size=seg_bs, shuffle=False, num_workers=2, pin_memory=True)

    seg_model = UNetSmall(in_ch=1, base=32).to(device)
    train_segmenter(seg_model, seg_train_loader, seg_val_loader, device, epochs=args.epochs_seg, lr=1e-3)
    evaluate_segmenter(seg_model, seg_val_loader, device, threshold=0.5)
    seg_model.eval()
    dummy_seg = torch.randn(1, 1, 256, 256, device=device)
    export_onnx_single_file(seg_model, dummy_seg, out_dir / "stroke_location_segmenter_single.onnx")



    print("\nAll done.")
    print("ONNX models saved to:", out_dir.resolve())
    print("Files:")
    print(" - stroke_type_classifier_single.onnx")
    print(" - stroke_location_segmenter_single.onnx")


if __name__ == "__main__":
    main()
