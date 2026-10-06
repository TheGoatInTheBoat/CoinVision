import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

# Load pre-trained ResNet50 weights
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)

# Replace the final fully connected layer to fit to my dataset
num_features = model.fc.in_features
num_classes = 10  # num categories
model.fc = nn.Linear(num_features, num_classes)

# Inference transforms by weight metadata
preprocess_transforms = weights.transforms()