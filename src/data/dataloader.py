import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_data_loaders(data_dir, image_size=224, batch_size=32, num_workers=4): #img size is set to 224 pixels, batch size 32 matab at a time 32 images ko process karega, num_workers 4 threads use karega data loading ke liye
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)), #resize input image to 224x224 pixels
        transforms.RandomHorizontalFlip(), #images are randomly flipped horizontally for data augmentation
        transforms.ToTensor(), #images to tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), #normalize the image on RGB channels each and these values are from ImageNet dataset
    ])
#loading the dataset
    train_dataset = datasets.ImageFolder(f'{data_dir}/Train', transform=train_transform)
    val_dataset = datasets.ImageFolder(f'{data_dir}/Validation', transform=test_transform)
    test_dataset = datasets.ImageFolder(f'{data_dir}/Test', transform=test_transform)
#data loaders banaye gaye hain acc to our needs
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
