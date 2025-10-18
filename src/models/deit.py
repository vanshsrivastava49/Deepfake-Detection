#Data-efficient image transformer model
#this is also pretrained on ImageNet dataset of 1Million images and 1000 objects like animals, objects etc.
import timm
import torch.nn as nn

def get_deit(num_classes=2):
    model = timm.create_model('deit_small_patch16_224', pretrained=True) #pretrained model classification head is modifiedwith new linear layer to have 2 output classes
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model

#This model takes images of size 224x224 pixels and then it is divided into patches of size 16x16 pixels each. Which means 224/16=14 patches along width and height each results in 14x14=196 patches.
#These patched are flattened into 1D vectors like representaion in numbers and then these vectors are passed through linear layers to create embeddings into transformer model. 
#The embeddings are accepted in a sequence and a token is added at the start of the sequence which helps in classification task.
#The sequence of patch embeddings in then pushed to transformer encoder stack and this layer uses self-attention mechanism to capture relationships between different patches in the image.
#after the relationship is captured the output corresponding to the classification token is extracted and passed through a final linear layer (head) to produce the class predictions (Fake or Real).


#as the transformer model can only understand through vectors(number representaion) that is why embeddings are created and passed to transformer model and the sequence of these embeddings represent the whole image.
#Vector is a list of numbers that represents some data in a format that machine learning models can understand and process.
#self-attention mechanism helps the model to focus on different parts of the input sequence and decide which parts are more important for making predictions.
#transformer model is a neural network architecture that is designed for language tasks but now for images they are also used that uses self-attention layer and feed forward layers
#feed forward layers is a fully connected neural network layer that processes the output of attention layer for each patch independently to capture complex patterns.