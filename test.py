"""
Unit Tests for CT Stroke image classification model using pytest fixtures

Tests cover: classifier model, segmenter model, data pipeline, mask generation (segmenter), 
model architecture, helper functions and dataset classes

Requires:
    pip install pytest numpy pillow torch torchvision onnxruntime
"""

import os
import sys
import zipfile
import tempfile
import pytest
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms


# import project modules from Training ONNX.py

import importlib.util

TRAINING_SCRIPT = Path(__file__).parent / "Training ONNX.py"

# only load if the training script exists 
if TRAINING_SCRIPT.exists():
    spec = importlib.util.spec_from_file_location("training_onnx", TRAINING_SCRIPT)
    training = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(training)
else:
    training = None



# fixtures: reusable test data
@pytest.fixture
def sample_rgb_image():
    # creates a dummy 512x512 RGB image (simulates a CT slice)
    arr = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def sample_gray_image():
    # creates a dummy 256x256 grayscale image
    arr = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    return Image.fromarray(arr, mode="L")


@pytest.fixture
def sample_png_dir(tmp_path):
    # creates a temp directory with 5 dummy PNG files
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(5):
        img = Image.fromarray(
            np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        )
        img.save(img_dir / f"slice_{i}.png")
    return img_dir


@pytest.fixture
def sample_zip(sample_png_dir, tmp_path):
    # creates a zip file containing PNG images
    zip_path = tmp_path / "test_scan.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for png in sample_png_dir.glob("*.png"):
            zf.write(png, png.name)
    return zip_path


@pytest.fixture
def overlay_pair(tmp_path):
    # creates an original + overlay image pair with a known red region
    # Original: gray brain-like image
    orig = np.full((256, 256, 3), 128, dtype=np.uint8)

    # overlay: same as original but with a red square (simulating hemorrhage)
    overlay = orig.copy()
    overlay[80:150, 80:150] = [255, 50, 50]  # red region

    orig_img = Image.fromarray(orig)
    over_img = Image.fromarray(overlay)

    orig_path = tmp_path / "scan.png"
    over_path = tmp_path / "scan_overlay.png"
    orig_img.save(orig_path)
    over_img.save(over_path)

    return orig_img, over_img, orig_path, over_path


@pytest.fixture
def onnx_model_dir():
    # returns path to onnx_out directory if it exists
    onnx_dir = Path(__file__).parent / "onnx_out"
    if not onnx_dir.exists():
        pytest.skip("onnx_out directory not found — skipping ONNX tests")
    return onnx_dir


# Tests that verify the classifier model works correctly:
class TestONNXClassifier:

    def _load_session(self, onnx_dir):
        import onnxruntime as ort
        model_path = onnx_dir / "stroke_type_classifier_single.onnx"
        if not model_path.exists():
            pytest.skip("Classifier ONNX file not found")
        return ort.InferenceSession(str(model_path))

    def test_classifier_loads(self, onnx_model_dir):
        # model file loads without errors
        session = self._load_session(onnx_model_dir)
        assert session is not None

    def test_classifier_input_shape(self, onnx_model_dir):
        # model expects [batch, 3, 224, 224] input
        session = self._load_session(onnx_model_dir)
        input_info = session.get_inputs()[0]
        assert input_info.shape[1:] == [3, 224, 224]

    def test_classifier_output_shape(self, onnx_model_dir):
        # model outputs 3 classes (Normal, Ischemic, Hemorrhagic)
        session = self._load_session(onnx_model_dir)
        dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
        outputs = session.run(None, {"input": dummy})
        assert outputs[0].shape == (1, 3)

    def test_classifier_output_varies(self, onnx_model_dir):
        # different inputs produce different outputs (model is not collapsed)
        session = self._load_session(onnx_model_dir)
        black = np.zeros((1, 3, 224, 224), dtype=np.float32)
        white = np.ones((1, 3, 224, 224), dtype=np.float32)
        out_black = session.run(None, {"input": black})[0]
        out_white = session.run(None, {"input": white})[0]
        assert not np.allclose(out_black, out_white), "Model gives same output for different inputs"

    def test_classifier_softmax_sums_to_one(self, onnx_model_dir):
        # softmax of logits sums to ~1.0 (valid probability distribution)
        session = self._load_session(onnx_model_dir)
        dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
        logits = session.run(None, {"input": dummy})[0]
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        assert abs(probs.sum() - 1.0) < 1e-5

    def test_classifier_batch_inference(self, onnx_model_dir):
        # model handles a batch of 4 images
        session = self._load_session(onnx_model_dir)
        batch = np.random.randn(4, 3, 224, 224).astype(np.float32)
        outputs = session.run(None, {"input": batch})
        assert outputs[0].shape == (4, 3)

    def test_classifier_deterministic(self, onnx_model_dir):
        # same input produces same output (deterministic inference)
        session = self._load_session(onnx_model_dir)
        dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
        out1 = session.run(None, {"input": dummy})[0]
        out2 = session.run(None, {"input": dummy})[0]
        np.testing.assert_array_equal(out1, out2)

# Tests that verify the segmenter model works correctly:
class TestONNXSegmenter:

    def _load_session(self, onnx_dir):
        import onnxruntime as ort
        model_path = onnx_dir / "stroke_location_segmenter_single.onnx"
        if not model_path.exists():
            pytest.skip("Segmenter ONNX file not found")
        return ort.InferenceSession(str(model_path))

    def test_segmenter_loads(self, onnx_model_dir):
        # model file loads without errors
        session = self._load_session(onnx_model_dir)
        assert session is not None

    def test_segmenter_input_shape(self, onnx_model_dir):
        # model expects [batch, 1, 256, 256] grayscale input
        session = self._load_session(onnx_model_dir)
        input_info = session.get_inputs()[0]
        assert input_info.shape[1:] == [1, 256, 256]

    def test_segmenter_output_shape(self, onnx_model_dir):
        # model outputs [batch, 1, 256, 256] mask
        session = self._load_session(onnx_model_dir)
        dummy = np.random.randn(1, 1, 256, 256).astype(np.float32)
        outputs = session.run(None, {"input": dummy})
        assert outputs[0].shape == (1, 1, 256, 256)

    def test_segmenter_output_is_logits(self, onnx_model_dir):
        # output contains both positive and negative values (raw logits, not probabilities)
        session = self._load_session(onnx_model_dir)
        dummy = np.random.randn(1, 1, 256, 256).astype(np.float32)
        output = session.run(None, {"input": dummy})[0]
        assert np.all(np.isfinite(output)), "Output contains non-finite values"
        assert output.max() != output.min(), "Output is constant — model may have collapsed"

    def test_segmenter_mask_binary_after_threshold(self, onnx_model_dir):
        # after sigmoid + threshold, output is a valid binary mask
        session = self._load_session(onnx_model_dir)
        dummy = np.random.randn(1, 1, 256, 256).astype(np.float32)
        logits = session.run(None, {"input": dummy})[0]
        probs = 1 / (1 + np.exp(-logits))  # sigmoid
        mask = (probs > 0.5).astype(np.uint8)
        unique_vals = set(np.unique(mask))
        assert unique_vals.issubset({0, 1})

    def test_segmenter_deterministic(self, onnx_model_dir):
        # same input produces same output
        session = self._load_session(onnx_model_dir)
        dummy = np.random.randn(1, 1, 256, 256).astype(np.float32)
        out1 = session.run(None, {"input": dummy})[0]
        out2 = session.run(None, {"input": dummy})[0]
        np.testing.assert_array_equal(out1, out2)

    def test_segmenter_blank_input_low_activation(self, onnx_model_dir):
        #A blank image should produce mostly negative logits (no stroke detected)."""
        session = self._load_session(onnx_model_dir)
        blank = np.zeros((1, 1, 256, 256), dtype=np.float32)
        logits = session.run(None, {"input": blank})[0]
        probs = 1 / (1 + np.exp(-logits))
        mask_pct = (probs > 0.5).mean()
        assert mask_pct < 0.5, f"Blank image triggered {mask_pct*100:.1f}% mask — expected mostly inactive"


#  Data pipeline tests
class TestDataPipeline:
    # tests for data loading, zip handling, and image utilities

    @pytest.mark.skipif(training is None, reason="Training script not found")
    def test_zip_extraction(self, sample_zip, tmp_path):
        # zip files are correctly extracted to output directory
        out_dir = tmp_path / "extracted"
        training.unzip_if_needed(sample_zip, out_dir)
        pngs = list(out_dir.rglob("*.png"))
        assert len(pngs) == 5

    @pytest.mark.skipif(training is None, reason="Training script not found")
    def test_zip_extraction_idempotent(self, sample_zip, tmp_path):
        # calling unzip twice doesn't cause errors
        out_dir = tmp_path / "extracted"
        training.unzip_if_needed(sample_zip, out_dir)
        training.unzip_if_needed(sample_zip, out_dir)  # second call
        pngs = list(out_dir.rglob("*.png"))
        assert len(pngs) == 5

    @pytest.mark.skipif(training is None, reason="Training script not found")
    def test_pil_open_rgb(self, sample_png_dir):
        # tests that pil_open_rgb returns an RGB image
        png = next(sample_png_dir.glob("*.png"))
        img = training.pil_open_rgb(png)
        assert img.mode == "RGB"

    @pytest.mark.skipif(training is None, reason="Training script not found")
    def test_pil_open_gray(self, sample_png_dir):
        # tests that pil_open_gray returns a grayscale image
        png = next(sample_png_dir.glob("*.png"))
        img = training.pil_open_gray(png)
        assert img.mode == "L"

    @pytest.mark.skipif(training is None, reason="Training script not found")
    def test_find_png_dir(self, sample_png_dir):
        # find_png_dir finds the directory containing PNGs
        found = training.find_png_dir(sample_png_dir.parent)
        pngs = list(found.glob("*.png"))
        assert len(pngs) > 0

    @pytest.mark.skipif(training is None, reason="Training script not found")
    def test_find_png_dir_raises_on_empty(self, tmp_path):
        #find_png_dir raises RuntimeError if no PNGs found
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(RuntimeError):
            training.find_png_dir(empty_dir)

    def test_classifier_transform_output_shape(self, sample_rgb_image):
        # classifier transform resizes to 224x224 and converts to tensor
        tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        tensor = tf(sample_rgb_image)
        assert tensor.shape == (3, 224, 224)

    def test_classifier_transform_value_range(self, sample_rgb_image):
        # ToTensor normalizes pixel values to [0, 1]
        tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        tensor = tf(sample_rgb_image)
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_segmenter_resize(self, sample_gray_image):
        # Grayscale image resizes correctly to 256x256
        resized = sample_gray_image.resize((256, 256), resample=Image.BILINEAR)
        assert resized.size == (256, 256)
        assert resized.mode == "L"



# Mask generation tests
class TestMaskGeneration:
    # tests for binary mask generation from overlay images

    @pytest.mark.skipif(training is None, reason="Training script not found")
    def test_mask_shape_matches_input(self, overlay_pair):
        # tests that generated mask has the same spatial dimensions as the input
        orig_img, over_img, _, _ = overlay_pair
        mask = training.build_binary_mask_from_overlay(orig_img, over_img)
        assert mask.shape == (256, 256)

    @pytest.mark.skipif(training is None, reason="Training script not found")
    def test_mask_is_binary(self, overlay_pair):
        # tests that mask contains only 0 and 1 values."""
        orig_img, over_img, _, _ = overlay_pair
        mask = training.build_binary_mask_from_overlay(orig_img, over_img)
        unique = set(np.unique(mask))
        assert unique.issubset({0, 1})

    @pytest.mark.skipif(training is None, reason="Training script not found")
    def test_mask_detects_overlay_region(self, overlay_pair):
        # mask is non-zero where the overlay differs from the original
        orig_img, over_img, _, _ = overlay_pair
        mask = training.build_binary_mask_from_overlay(orig_img, over_img)
        assert mask.sum() > 0, "Mask should detect the red overlay region"

    @pytest.mark.skipif(training is None, reason="Training script not found")
    def test_mask_region_location(self, overlay_pair):
        # mask activation is concentrated in the region where overlay was placed
        orig_img, over_img, _, _ = overlay_pair
        mask = training.build_binary_mask_from_overlay(orig_img, over_img, cleanup=False)
        # The red region was placed at [80:150, 80:150]
        region_activation = mask[80:150, 80:150].mean()
        outside_activation = mask[:80, :80].mean()
        assert region_activation > outside_activation

    @pytest.mark.skipif(training is None, reason="Training script not found")
    def test_identical_images_produce_empty_mask(self):
        # two identical images should produce an empty (all-zero) mask
        img = Image.fromarray(np.full((100, 100, 3), 128, dtype=np.uint8))
        mask = training.build_binary_mask_from_overlay(img, img)
        assert mask.sum() == 0

    @pytest.mark.skipif(training is None, reason="Training script not found")
    def test_mask_with_no_cleanup(self, overlay_pair):
        # mask generation works with cleanup disabled
        orig_img, over_img, _, _ = overlay_pair
        mask = training.build_binary_mask_from_overlay(
            orig_img, over_img, cleanup=False
        )
        assert mask.shape == (256, 256)
        assert mask.sum() > 0



# Model architecture tests
class TestModelArchitecture:

    @pytest.mark.skipif(training is None, reason="Training script not found")
    def test_classifier_forward_pass(self):
        # tests classifier produces correct output shape
        model = training.make_resnet18_classifier(num_classes=3, pretrained=False)
        model.eval()
        dummy = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (2, 3)

    @pytest.mark.skipif(training is None, reason="Training script not found")
    def test_classifier_output_num_classes(self):
        # classifier final layer has 3 output neurons
        model = training.make_resnet18_classifier(num_classes=3, pretrained=False)
        assert model.fc.out_features == 3

    @pytest.mark.skipif(training is None, reason="Training script not found")
    def test_unet_forward_pass(self):
        # tests segmenter produces correct output shape
        model = training.UNetSmall(in_ch=1, base=32)
        model.eval()
        dummy = torch.randn(2, 1, 256, 256)
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (2, 1, 256, 256)



#  Tests for helper functions used during training
class TestTrainingUtilities:

    @pytest.mark.skipif(training is None, reason="Training script not found")
    def test_split_indices_no_overlap(self):
        # tests train and val sets have no overlapping indices
        train_idx, val_idx = training.split_indices(100, val_frac=0.15, seed=42)
        assert len(set(train_idx) & set(val_idx)) == 0

    @pytest.mark.skipif(training is None, reason="Training script not found")
    def test_split_indices_reproducible(self):
        # same seed produces the same split
        t1, v1 = training.split_indices(100, val_frac=0.15, seed=42)
        t2, v2 = training.split_indices(100, val_frac=0.15, seed=42)
        assert t1 == t2
        assert v1 == v2

    @pytest.mark.skipif(training is None, reason="Training script not found")
    def test_split_indices_different_seeds(self):
        # different seeds produce different splits."""
        t1, _ = training.split_indices(100, val_frac=0.15, seed=42)
        t2, _ = training.split_indices(100, val_frac=0.15, seed=99)
        assert t1 != t2

    @pytest.mark.skipif(training is None, reason="Training script not found")
    def test_split_indices_val_fraction(self):
        # validation set size is approximately correct
        _, val_idx = training.split_indices(1000, val_frac=0.2, seed=42)
        assert 190 <= len(val_idx) <= 210  # ~200 with some rounding tolerance

    @pytest.mark.skipif(training is None, reason="Training script not found")
    def test_dice_loss_perfect_prediction(self):
        # dice loss is ~0 when prediction perfectly matches target
        logits = torch.full((1, 1, 64, 64), 10.0)  # high confidence positive
        target = torch.ones(1, 64, 64)
        loss = training.dice_loss_from_logits(logits, target)
        assert loss.item() < 0.05

    @pytest.mark.skipif(training is None, reason="Training script not found")
    def test_dice_loss_worst_prediction(self):
        # dice loss is ~1 when prediction is completely wrong
        logits = torch.full((1, 1, 64, 64), -10.0)  # high confidence negative
        target = torch.ones(1, 64, 64)
        loss = training.dice_loss_from_logits(logits, target)
        assert loss.item() > 0.9

    @pytest.mark.skipif(training is None, reason="Training script not found")
    def test_dice_loss_range(self):
        # dice loss is always between 0 and 1."""
        for _ in range(10):
            logits = torch.randn(2, 1, 32, 32)
            target = (torch.rand(2, 32, 32) > 0.5).float()
            loss = training.dice_loss_from_logits(logits, target)
            assert 0.0 <= loss.item() <= 1.0



#  Tests for dataset classes
class TestDatasets:

    @pytest.mark.skipif(training is None, reason="Training script not found")
    def test_classification_dataset_output_types(self, sample_png_dir):
        # dataset returns (tensor, label) tuple with correct types
        ds = training.StrokeClassificationDataset(
            normal_dir=sample_png_dir,
            ischemic_dir=sample_png_dir,
            hemorr_dir=sample_png_dir,
            transform=transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]),
        )
        img, label = ds[0]
        assert isinstance(img, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert img.shape == (3, 224, 224)
        assert label.dtype == torch.long

    @pytest.mark.skipif(training is None, reason="Training script not found")
    def test_classification_dataset_labels(self, sample_png_dir):
        # dataset assigns correct labels: 0=Normal, 1=Ischemic, 2=Hemorrhagic."""
        ds = training.StrokeClassificationDataset(
            normal_dir=sample_png_dir,
            ischemic_dir=sample_png_dir,
            hemorr_dir=sample_png_dir,
            transform=transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]),
        )
        labels = [ds[i][1].item() for i in range(len(ds))]
        assert set(labels) == {0, 1, 2}

    @pytest.mark.skipif(training is None, reason="Training script not found")
    def test_classification_dataset_raises_on_empty(self, tmp_path):
        # dataset raises RuntimeError when directories have no images
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(RuntimeError):
            training.StrokeClassificationDataset(
                normal_dir=empty, ischemic_dir=empty, hemorr_dir=empty
            )

    @pytest.mark.skipif(training is None, reason="Training script not found")
    def test_subset_dataset(self, sample_png_dir):
        # SubsetDataset correctly maps indices
        ds = training.StrokeClassificationDataset(
            normal_dir=sample_png_dir,
            ischemic_dir=sample_png_dir,
            hemorr_dir=sample_png_dir,
            transform=transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]),
        )
        subset = training.SubsetDataset(ds, [0, 2, 4])
        assert len(subset) == 3

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])