# Deepfake Detection Model

This repository contains an deepfake detection model that trains and evaluates EfficientNet and DeiT models to classify human face images into two classes: **Fake** and **Real**. It includes modular code for training, evaluation, and inference on single or multiple images.

---
.
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

Results:- Accuracy -- EfficientNet - 93%     DeiT - 94.4%  
Dataset size:- ~1,40,000 images