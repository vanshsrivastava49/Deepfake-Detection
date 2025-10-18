import torch
from src.data.dataloader import get_data_loaders
from src.models.efficientnet import get_effnet
from src.models.deit import get_deit
from src.config import hyperparams as hp

def test_model_acc(model_name='efficientnet'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, _, test_loader = get_data_loaders(hp.data_dir, hp.image_size, hp.batch_size)

    if model_name.lower() == 'efficientnet':
        model = get_effnet(num_classes=2)
        model_path = 'saved_models/efficientnet_model_final.pth'
    elif model_name.lower() == 'deit':
        model = get_deit(num_classes=2)
        model_path = 'saved_models/deit_model_final.pth'
    else:
        raise ValueError("Invalid model_name. Choose 'efficientnet' or 'deit'.")

    # Load saved weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Import evaluate_model here to avoid circular imports if any
    from src.models.evaluator import evaluate_model

    acc = evaluate_model(model, test_loader, device=device)
    print(f"{model_name.capitalize()} Test Accuracy: {acc:.4f}")

if __name__ == '__main__':
    test_model_acc('efficientnet')
    test_model_acc('deit')