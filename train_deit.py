import torch
import torch.nn as nn
import timm
from utils import get_loaders

def main():
    data_dir = 'data/Dataset'
    batch_size = 32
    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_loaders(data_dir, batch_size)

    model = timm.create_model('deit_small_patch16_224', pretrained=True)
    model.head = nn.Linear(model.head.in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = torch.argmax(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        print(f'Epoch [{epoch+1}/{num_epochs}] Validation Accuracy: {correct/total:.4f}')

    # Save model
    torch.save(model.state_dict(), 'deit_model.pt')
    # Test
    print('Testing DeiT...')
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print(f'Test Accuracy (DeiT): {correct/total:.4f}')

if __name__ == "__main__":
    main()
