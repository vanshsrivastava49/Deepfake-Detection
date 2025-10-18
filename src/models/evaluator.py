#code to evaluate the model on test dataset
import torch
from tqdm import tqdm #progress bar for loops

def evaluate_model(model, test_loader, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #cuda bhaiya ki jai ho
    model = model.to(device)
    model.eval() #model set to evaluation mode
    correct = 0 #variable to count correct predictions
    total = 0 #total number of test samples
    with torch.no_grad(): #no need to compute gradients during evaluation(Gradients are analyzes loss function during training to update model weights to reduce loss)
        test_iter = tqdm(test_loader, desc="Testing")
        for imgs, labels in test_iter:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True) #non blocking to speed up data transfer b/w cpu and gpu
            outputs = model(imgs) #images are passed through model to get outputs
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item() #correct+1 if prediction matches label
            total += labels.size(0)
    acc = correct / total
    print(f"Test Accuracy: {acc:.4f}")
    return acc
