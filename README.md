# Stroke Detection
This project uses various supervised deep learning to predict if the patient is suffering from an ischemic or hemorrhagic stroke or neither. It then adds a highlighted area over any lesions in the brain

## Important Notice
**DO NOT UNDER ANY CIRCUMSTANCE USE THIS PROGRAM TO MAKE MEDICAL DECISIONS.**

This software is provided for educational and research purposes only. We take no responsibility or liability for any errors, inaccuracies, or outcomes resulting from the use of this program.

**THIS PROGRAM MUST NEVER BE USED IN ANY MEDICAL CONTEXTS**

## Model Training Overview
The system is trained using the following dataset with a 80/20 split: https://www.kaggle.com/datasets/ozguraslank/brain-stroke-ct-dataset/data into the two following models:

### Classification Model (Stroke Prediction)
The classification model is based on a ResNet-18 architecture.
#### Model Overview
- Architecture: ResNet-18
- Input: 224 x 224 RGB CT scan slices
- Output: 3-class logits
  - Normal
  - Ischemic Stroke
  - Hemorrhagic Stroke

### Segmentation Model (Leission Mask Creation)
The segementation model is based on a U-Net architecture.
#### Model Overview
- Architecture: U-Net
- Input: 256 x 256 grayscale images
- Output: Binary Mask (probability map with a percentage change of lesions at each pixel)

## Model Deployment
The models are made into ONNX files which are read in the flutter program and can be found at the onnx export files.
### Image Input
Images are inputed as zip files and are then preprocessed by both models:
#### Classification Preprocessing
- Images are resized to 224 x 224
- Images are converted to RGB
- Pixel values are normalized to [0,1]
#### Segmentation Preprocessing
- Images are resized to 256 x 256
- Images are converted to grayscale
- Pixel values are normalized
- Only images with leissions present will have a mask placed on them
  
## Running the App
To run the program, run the Stroke Detection Installer, you can choose to make it a desktop app or not.

## Authors
Owen Batstone
Nicholas Roy
Andrew Holt-Hindle

## MIT License
Copyright (c) 2026 
Owen Batstone, Nicholas Roy, Andrew Holt-Hindle

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
