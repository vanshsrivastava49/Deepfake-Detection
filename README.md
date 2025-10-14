# Image Classification Project

This repository contains an image classification project that trains and evaluates EfficientNet and DeiT models to classify images into two classes: **Fake** and **Real**. It includes modular code for training, evaluation, and inference on single or multiple images.

---

## Project Structure

image_classification_project/
│
├── data/ # Dataset folder (Train, Validation, Test)
│
├── saved_models/ # Saved trained model weights
│
├── src/
│   ├── config/
│   │   └── hyperparams.py # Hyperparameter settings
│   ├── data/
│   │   └── dataloader.py # Data loading utilities
│   ├── models/
│   │   ├── efficientnet.py # EfficientNet model definition
│   │   ├── deit.py # DeiT model definition
│   │   ├── trainer.py # Training loop
│   │   ├── evaluator.py # Evaluation loop
│
├── main.py # Training and evaluation script
├── test_image.py # Single image inference script
├── test_multiple_images.py # Multiple images inference script
├── requirements.txt # Required packages
└── README.md # Project overview and instructions