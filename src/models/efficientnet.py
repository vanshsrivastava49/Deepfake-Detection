#this is the code to return an EfficientNet model
#efficientnet model is trained on ImageNet dataset of 1Million images and 1000 objects like animals, objects etc.
import timm #timm libraray has the efficientnet-b0 pretrained model
import torch.nn as nn

def get_effnet(num_classes=2): #two classes: Fake and Real
    model = timm.create_model('efficientnet_b0', pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes) #model is modified to have 2 output classes
    return model

#the image is passed through a convolutional layer to extract low level features like edges, textures etc.