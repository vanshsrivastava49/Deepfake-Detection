import os
from PIL import Image
import torch
from torchvision import transforms
import timm

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Folder containing images to test
    image_folder = './test-images/'

    # images are resized to 224x224 pixels and then normalized
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Load model architecture
    model = timm.create_model('efficientnet_b0', pretrained=False)
    model.classifier = torch.nn.Linear(model.classifier.in_features, 2)

    # Load trained weights
    model.load_state_dict(torch.load('efficientnet_model_final.pth', map_location=device))
    model = model.to(device)
    model.eval()

    class_map = {0: 'Fake', 1: 'Real'}

    # Iterate over images in folder
    for img_name in os.listdir(image_folder):
        # Full image path
        img_path = os.path.join(image_folder, img_name)

        # Skip non-image files (optional)
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            continue

        # Load and preprocess image
        img = Image.open(img_path).convert('RGB')
        img_tensor = test_transform(img).unsqueeze(0).to(device) #add batch dimension to accept images in batches and not just one image

        # Prediction
        with torch.no_grad():
            output = model(img_tensor)
            prob = torch.softmax(output, dim=1)
            pred_class = torch.argmax(prob, dim=1).item()

        print(f"{img_name} -> Prediction: {class_map[pred_class]}, Probability: {prob[0, pred_class].item():.4f}")

if __name__ == '__main__':
    main()
