# MNIST CNN Classifier in PyTorch 🧠🔥

This project demonstrates a step-by-step implementation of a Convolutional Neural Network (CNN) using **PyTorch** to classify handwritten digits from the **MNIST** dataset.

It includes:
- A custom CNN model
- Real-time accuracy, precision, and recall evaluation
- Support for uploading and predicting custom digit images
- Saving and downloading trained models

---

## 🚀 Features

- 📦 Clean CNN architecture (3 conv layers, pooling, fully connected)
- 📊 Metrics with `torchmetrics`: Accuracy, Precision, Recall
- 🎨 Custom digit prediction with preprocessing & visualization
- 💾 Save trained model to Google Drive
- 🔽 Download model weights directly from Colab

---

## 🛠️ Libraries Used

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torchmetrics import Accuracy, Precision, Recall
from tqdm import tqdm


Training
Dataset: MNIST

Batch Size: 32

Epochs: 10

Optimizer: Adam

Loss Function: CrossEntropyLoss

Early Saving: Saves the model with the lowest loss

📸 Predict Your Own Digit
Upload a .png image of a digit written in thin or thick stroke, and run the prediction section to get the model's guess.

Ensure:

Size is 28x28

Background is light, digit is dark

💾 Save and Download Model
python
Copy
Edit
# Save
torch.save(model.state_dict(), "model.pt")

# Load
model.load_state_dict(torch.load("model.pt"))
📂 Folder Structure
bash
Copy
Edit
mnist-cnn-pytorch/
│
├── train_and_eval.py
├── predict_digit.py
├── requirements.txt
├── README.md
└── /dataset
