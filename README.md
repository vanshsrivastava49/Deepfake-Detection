# Deepfake Detection Model

An advanced deepfake detection system using EfficientNet and DeiT (Vision Transformer) architectures to classify human face images as **Fake** or **Real**. This repository provides modular code for data loading, training, evaluation, and inference on both single and multiple images.

---

## Important Key Features

- **Model Architectures:** Implements both EfficientNet and DeiT (Vision Transformer) models for robust face classification.
- **Training & Evaluation:** Includes scripts for training models, evaluating on test datasets, and testing on custom images.
- **Modular Code:** Clean, organized, and scalable codebase for easy experimentation and extension.
- **Batch & Single Image Testing:** Supports inference on a batch of images or a single image with detailed results.

---

## Project Structure

```
├── saved_models/               # Directory where trained models are saved
│   ├── efficientnet_model_final.pth   # Trained EfficientNet model
│   └── deit_model_final.pth           # Trained DeiT model
├── src/                        # Source code for data loading, training, and evaluation
│   ├── data/                   # Data loading and preprocessing scripts
│   │   └── dataloader.py       # DataLoader configuration and transforms
│   ├── models/                 # Model architecture and training scripts
│   │   ├── trainer.py          # Training code
│   │   ├── evaluator.py        # Evaluation code
│   │   ├── efficientnet.py     # EfficientNet architecture
│   │   └── deit.py             # DeiT architecture
│   └── config/                 # Hyperparameter configuration
│       └── hyperparams.py      # Hyperparameters like learning rate, batch size, epochs etc.
├── test-images/                # Folder containing images for inference (for testing the model)
├── main.py                     # Main script for training and testing the models
├── test_model_acc.py           # code to know the accuracy of the both deit and efficienet model on test dataset
├── test.py                     # test the model on multiple images of our own
└── README.md                   # This file (project overview, setup, results, etc.)
```

---

## Results

- **EfficientNet Test Accuracy:** 93%
- **DeiT Test Accuracy:** 94.4%
- **Dataset Size:** ~140,000 images (balanced real and fake faces)

---
