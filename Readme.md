# CIFAR-10 Image Classification using CNN

## TASK 1 - Part A: Basic CNN Model Implementation

###  Project Description
This project implements an image classification system using a **Convolutional Neural Network (CNN)** to classify images from the **CIFAR-10 dataset**. The model is built using **TensorFlow** and **Keras** frameworks.

The CIFAR-10 dataset contains **60,000 color images** across **10 different classes**:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

###  Objective
The main goal of this project is to:
- Build a CNN model from scratch.
- Train the model on the CIFAR-10 dataset.
- Evaluate the model's performance using Accuracy.
- Predict and display sample images with their class labels.

---
## Folder Structure
```
CNN_CIFAR10_PROJECT
│
├─ dataset_preprocessing.py   # Data Loading and Preprocessing
├─ train_model.py            # Model Building and Training
├─ evaluate_model.py         # Model Evaluation
├─ test_model.py             # Model Testing with Image Predictions
├─ requirements.txt          # Libraries Required
└─ README.md                 # Project Documentation
```
---

## 1.  Dataset Preprocessing
### File: `dataset_preprocessing.py`
This file loads and preprocesses the CIFAR-10 dataset.

#### Preprocessing Steps:
- Load dataset using `cifar10.load_data()`
- Normalize the image pixels by dividing by 255.0
- One-hot encode labels using `to_categorical()`

---

## 2.  Model Training
### File: `train_model.py`
In this file, the CNN model is built and trained.

#### Model Architecture:
| Layer               | Filters | Kernel Size | Activation | Purpose              |
|-------------------|-------|------------|-----------|-------------------|
| Conv2D           | 32    | (3x3)     | ReLU      | Feature Extraction |
| MaxPooling2D     | -     | (2x2)     | -         | Downsampling      |
| Conv2D           | 64    | (3x3)     | ReLU      | Feature Extraction |
| MaxPooling2D     | -     | (2x2)     | -         | Downsampling      |
| Flatten          | -     | -         | -         | Convert Matrix to Vector |
| Dense            | 512   | -         | ReLU      | Fully Connected Layer |
| Dense            | 10    | -         | Softmax   | Output Layer (10 Classes) |

---

### Training Configuration:
- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Epochs: 10
- Batch Size: 64

#### Training Command
```bash
python train_model.py
```
---

## 3.  Model Evaluation
### File: `evaluate_model.py`
This file evaluates the trained model on the CIFAR-10 **Test Dataset**.

Command to Run:
```bash
python evaluate_model.py
```
Example Output:
```
Test Accuracy: 71.20%
```
---

## 4.  Image Prediction
### File: `test_model.py`
This file predicts the first 10 images from the CIFAR-10 test set and displays the images with their predicted labels.

#### Command to Run:
```bash
python test_model.py
```
Example Output:
```
Model Loaded Successfully!
 Predicted: Airplane |  Actual: Airplane
 Predicted: Frog     |  Actual: Frog
...
```
---

## 5.  Installation Guide
### Prerequisites
- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib

### Install Dependencies
```bash
pip install -r requirements.txt
```
---

##  Results
| Metric   | Score |
|----------|-------|
| Accuracy | 71.20% |

---

##  How to Run the Project
1. Clone the Repository:
```bash
git clone https://github.com/YourUsername/CNN_CIFAR10_PROJECT.git
```
2. Install Dependencies:
```bash
pip install -r requirements.txt
```
3. Train the Model:
```bash
python train_model.py
```
4. Evaluate the Model:
```bash
python evaluate_model.py
```
5. Test Predictions:
```bash
python test_model.py
```
---

##  Documentation
All files and code are explained in the **Documentation Folder**.

---
##  Future Improvements
- Add Dropout Layers
- Implement Batch Normalization
- Increase Number of Layers
- Try Different Optimizers
