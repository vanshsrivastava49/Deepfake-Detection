import timm
import torch.nn as nn

def get_effnet(num_classes=2):
    model = timm.create_model('efficientnet_b0', pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model
