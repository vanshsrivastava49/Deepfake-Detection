import torch
from src.data.dataloader import get_data_loaders
from src.models.trainer import train_model
from src.models.evaluator import evaluate_model
from src.models.efficientnet import get_effnet
from src.models.deit import get_deit
import src.config.hyperparams as hp

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_data_loaders(hp.data_dir, hp.image_size, hp.batch_size)

    # EfficientNet
    effnet_model = get_effnet(num_classes=2)
    print("Training EfficientNet...")
    effnet_model = train_model(effnet_model, train_loader, val_loader, epochs=hp.num_epochs, lr=hp.learning_rate, device=device)
    torch.save(effnet_model.state_dict(), 'saved_models/efficientnet_model_final.pth')
    evaluate_model(effnet_model, test_loader, device=device)

    # DeiT
    deit_model = get_deit(num_classes=2)
    print("Training DeiT...")
    deit_model = train_model(deit_model, train_loader, val_loader, epochs=hp.num_epochs, lr=hp.learning_rate, device=device)
    torch.save(deit_model.state_dict(), 'saved_models/deit_model_final.pth')
    evaluate_model(deit_model, test_loader, device=device)
