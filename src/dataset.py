import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_dataloaders(data_dir, img_size=224, batch_size=16):
    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    test_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "Train"), transform=train_tfms)
    val_ds   = datasets.ImageFolder(os.path.join(data_dir, "Validation"), transform=test_tfms)
    test_ds  = datasets.ImageFolder(os.path.join(data_dir, "Test"), transform=test_tfms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, len(train_ds.classes)
