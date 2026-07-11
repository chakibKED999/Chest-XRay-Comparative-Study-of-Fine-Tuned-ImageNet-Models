# 🩻 Comparative Study of Fine-Tuned ImageNet Models for Chest X-Ray Pneumonia Detection

> A comprehensive deep learning comparative study of multiple ImageNet-pretrained convolutional neural networks for automatic pneumonia detection from chest X-ray images using transfer learning, fine-tuning, and extensive performance evaluation.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-red)
![Torchvision](https://img.shields.io/badge/Torchvision-Models-orange)
![TransferLearning](https://img.shields.io/badge/Transfer-Learning-green)
![MedicalAI](https://img.shields.io/badge/Medical-AI-success)
![Master](https://img.shields.io/badge/Master-MIV-lightgrey)

---

# 📖 Overview

This project presents a comprehensive **comparative study** of several state-of-the-art **ImageNet-pretrained Convolutional Neural Networks (CNNs)** for **binary chest X-ray image classification**.

The objective is to automatically determine whether a chest radiograph belongs to a patient with **Pneumonia** or a **Normal** healthy lung.

Instead of evaluating a single neural network architecture, this work compares five widely used pretrained models under identical experimental conditions in order to identify the architecture that provides the best compromise between classification accuracy, computational efficiency, and generalization capability.

Each network is trained using **Transfer Learning**, followed by **Fine-Tuning**, and evaluated using multiple learning rates and comprehensive classification metrics.

This project was developed as part of the **Master's Program in Image Processing & Artificial Intelligence (MIV)** at **USTHB**.

---

# ✨ Features

- 🩻 Automatic chest X-ray classification
- 🧠 Comparison of five pretrained CNN architectures
- 🔄 Transfer Learning
- 🎯 Fine-Tuning
- 📊 Multi-learning-rate experimentation
- 📈 Training and validation visualization
- 📉 Confusion Matrix generation
- 📋 Classification Reports
- 📊 Accuracy, Precision, Recall and F1-score evaluation
- ⚖️ Class weight balancing
- 💾 Automatic best model checkpoint saving
- 🖼️ Misclassified image visualization
- 📑 Automatic CSV and performance summary generation

---

# 📑 Project Report

## Project Objective

Pneumonia is one of the leading causes of respiratory disease worldwide and represents a major public health concern. Early diagnosis significantly improves treatment outcomes and reduces mortality.

Medical image interpretation, however, requires experienced radiologists and can be time-consuming, particularly in healthcare systems with limited resources.

The objective of this project is to develop and compare several deep learning models capable of automatically classifying chest X-ray images into two categories:

- 🫁 Normal
- 🦠 Pneumonia

Rather than focusing on a single architecture, this project evaluates multiple pretrained CNN models to determine which architecture performs best on the same dataset using identical training procedures.

---

# 🧠 Compared Models

This comparative study evaluates the following ImageNet-pretrained architectures:

- VGG16
- MobileNetV3-Large
- ResNet50
- InceptionV3
- EfficientNet-B0

Each architecture represents a different generation of convolutional neural networks, offering unique design philosophies and performance characteristics.

---

# 🔬 Why Compare Multiple Models?

Choosing the appropriate neural network architecture is one of the most important decisions in any computer vision project.

Different CNN architectures differ in:

- Model complexity
- Number of parameters
- Computational cost
- Training speed
- Generalization capability
- Accuracy

By comparing several pretrained models under identical conditions, this project identifies the architecture that achieves the best balance between prediction performance and computational efficiency.

This methodology also provides valuable insights into how modern CNN architectures perform on medical imaging tasks.

---

# 🔄 Why Transfer Learning?

Training deep convolutional neural networks from scratch requires enormous datasets and extensive computational resources.

Instead, this project leverages **Transfer Learning** using models pretrained on the **ImageNet** dataset.

The pretrained models already possess rich visual representations learned from millions of images, including:

- Edges
- Corners
- Textures
- Shapes
- Anatomical patterns
- High-level visual features

Only the final classification layers are adapted to the chest X-ray classification task.

Transfer Learning offers several advantages:

- Faster convergence
- Less training time
- Better generalization
- Reduced overfitting
- Higher performance on limited datasets

---

# 🎯 Why Fine-Tuning?

Although pretrained models already contain powerful visual representations, medical imaging differs significantly from natural photographs.

Fine-tuning allows the deeper convolutional layers to gradually adapt their learned features to chest X-ray imagery.

Compared to feature extraction alone, fine-tuning generally improves:

- Classification accuracy
- Feature specialization
- Robustness
- Generalization

---

# ⚙️ Training Strategy

Each architecture follows the same two-stage training strategy.

## Stage 1 — Feature Extraction

The pretrained backbone is frozen.

Only the final classification layer is trained to learn the Normal/Pneumonia categories.

---

## Stage 2 — Fine-Tuning

The pretrained backbone is partially or fully unfrozen.

Training continues using a smaller learning rate, allowing the network to adapt its learned features to medical images while preserving the knowledge acquired from ImageNet.

---

# 📊 Evaluation Strategy

To ensure a fair comparison, every model is evaluated using identical performance metrics.

The evaluation includes:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Classification Report
- Learning Curves

Multiple learning rates are tested for every architecture to identify the optimal training configuration.

The best-performing model is automatically selected based on validation performance.

---

# 🏗️ Pipeline

## 1️⃣ Dataset Preparation

- Load chest X-ray dataset
- Separate Normal and Pneumonia classes
- Create training, validation and test datasets
- Verify dataset integrity

---

## 2️⃣ Data Preprocessing

Images are resized according to each model's required input resolution.

Normalization uses the standard ImageNet statistics.

Training augmentation includes:

- Random Horizontal Flip
- Random Rotation
- Random Affine Transformation
- Normalization

These augmentations improve model robustness and reduce overfitting.

---

## 3️⃣ Model Construction

The project evaluates the following pretrained architectures:

```python
VGG16
MobileNetV3-Large
ResNet50
InceptionV3
EfficientNet-B0
```

Each model replaces its original ImageNet classifier with a custom binary classification head.

---

## 4️⃣ Model Training

Every architecture is trained using:

- Transfer Learning
- Fine-Tuning
- Cross-Entropy Loss
- Adam Optimizer
- Learning Rate Scheduling
- Early Stopping
- Class Weight Balancing

The best model checkpoint is automatically saved during training.

---

## 5️⃣ Comparative Analysis

For every model and learning rate, the project records:

- Training Accuracy
- Validation Accuracy
- Test Accuracy
- Precision
- Recall
- F1-score
- Loss Curves
- Confusion Matrix
- Classification Report
- Training Time

All results are exported to CSV files for further analysis.

---

## 6️⃣ Error Analysis

To better understand model behavior, the notebook displays misclassified chest X-ray images for the best-performing configuration of each architecture.

This qualitative analysis helps identify common failure cases and provides insight into model limitations.

---

# 📂 Project Structure

```text
Chest-XRay-Comparative-Study/

│
├── chest_xray_comparative_study_pretrained_models_results.ipynb
│
├── models/
│   ├── best_vgg16.pth
│   ├── best_resnet50.pth
│   ├── best_mobilenetv3.pth
│   ├── best_inceptionv3.pth
│   └── best_efficientnetb0.pth
│
├── figures/
│   ├── learning_curves/
│   ├── confusion_matrices/
│   ├── comparison_plots/
│   └── misclassified_images/
│
├── csv/
│   ├── comparison_results.csv
│   ├── best_models.csv
│   └── evaluation_metrics.csv
│
├── study_summary.txt
│
└── README.md
```

---

# ▶️ Running the Project

Install the required packages

```bash
pip install torch torchvision matplotlib pandas numpy scikit-learn pillow seaborn
```

Launch Jupyter Notebook

```bash
jupyter notebook chest_xray_comparative_study_pretrained_models_results.ipynb
```

or open the notebook in Google Colab.

---

# 📈 Experimental Highlights

The study systematically compares each architecture using identical experimental settings, enabling a fair evaluation of:

- Classification performance
- Generalization capability
- Computational efficiency
- Sensitivity to learning rate
- Training stability

This comparative methodology provides practical guidance for selecting an appropriate CNN architecture for medical image classification tasks.

---

# ✅ Strengths

- Comparison of multiple state-of-the-art CNN architectures
- Transfer Learning with ImageNet weights
- Fine-Tuning strategy
- Multiple learning rate experiments
- Comprehensive evaluation metrics
- Automatic checkpoint saving
- Misclassified image analysis
- Exportable performance summaries
- Reproducible experimental pipeline

---

# ⚠️ Limitations

- Binary classification only
- Performance depends on dataset quality
- Requires GPU for efficient training
- Limited to chest X-ray images
- Medical diagnosis should always be confirmed by healthcare professionals

---

# 🚀 Future Improvements

- Multi-class thoracic disease classification
- Vision Transformer (ViT) comparison
- ConvNeXt and DenseNet evaluation
- Ensemble learning
- Explainable AI using Grad-CAM
- Automatic hyperparameter optimization
- Cross-validation experiments
- ONNX and TensorRT deployment
- Clinical decision support integration

---

# 🛠️ Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Pillow
- Google Colab

---

# 📚 References

- Simonyan & Zisserman — Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG16)
- He et al. — Deep Residual Learning for Image Recognition (ResNet)
- Howard et al. — Searching for MobileNetV3
- Szegedy et al. — Rethinking the Inception Architecture for Computer Vision
- Tan & Le — EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
- ImageNet Large Scale Visual Recognition Challenge
- PyTorch Documentation
- Torchvision Documentation
- Scikit-learn Documentation
