import torch, torch.nn as nn, torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from dataset import get_dataloaders
from models.cnn_a import CNN_A
from models.cnn_b import CNN_B
from models.cvit import CViT
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_model(model, train_loader, val_loader, epochs=20, lr=1e-4, name="model"):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    writer = SummaryWriter(f"runs/{name}")

    for epoch in range(epochs):
        model.train()
        total_loss, correct = 0, 0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
        acc = correct / len(train_loader.dataset)
        writer.add_scalar("Loss/train", total_loss/len(train_loader), epoch)
        writer.add_scalar("Acc/train", acc, epoch)

        # Validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                correct += (outputs.argmax(1) == labels).sum().item()
        val_acc = correct / len(val_loader.dataset)
        writer.add_scalar("Acc/val", val_acc, epoch)
        print(f"Epoch {epoch+1}: TrainAcc={acc:.3f} ValAcc={val_acc:.3f}")
    torch.save(model.state_dict(), f"outputs/{name}.pth")

if __name__ == "__main__":
    train_loader, val_loader, test_loader, num_classes = get_dataloaders(
    r"D:\MY-PROJECTS\Deepfake Detection\data\Dataset",
    img_size=224,
    batch_size=16
)
    model = CNN_A(num_classes=num_classes)   # swap with CNN_B or CViT
    train_model(model, train_loader, val_loader, epochs=10, name="cnn_a")
