#training code
import torch
import torch.nn as nn
import copy
from tqdm import tqdm

def train_model(model, train_loader, val_loader, epochs=10, lr=1e-4, device=None): #model=model jo train karna hai, train_loader(training images ko load karega) and val_loader(training images ko load karega), epcoh=number of times to iterate over entire training dataset, lr=learning rate(0.0001) 
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss() #loss function for multi class classification(fake and real)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) #adam optimizer to update model weights based on computed gradients

    best_model_wts = copy.deepcopy(model.state_dict()) #to save best model weights
    best_val_acc = 0.0 #variable to track best validation accuracy

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training")
        for imgs, labels in train_iter:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad() #clears the gradients from the previous step
            outputs = model(imgs)
            loss = criterion(outputs, labels)#to find the loss between predicted outputs and true labels
            loss.backward() #computes the gradients
            optimizer.step() #update the model parameters using computed gradients
            running_loss += loss.item() * imgs.size(0) #loss for each batch
            train_iter.set_postfix(loss=loss.item()) #display loss for current batch
        epoch_loss = running_loss / len(train_loader.dataset) #average loss for the epoch

#to find the validation accuracy after each epoch
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            val_iter = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation")
            for imgs, labels in val_iter:
                imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(imgs)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total #validation accuracy

        print(f"Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.4f} Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc: #if current validation accuracy is better than best so far
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, 'saved_models/best_model.pth') #save the best model weights
            print("Best model saved.")

    print(f"Training complete. Best Val Acc: {best_val_acc:.4f}")
    model.load_state_dict(best_model_wts)
    return model
