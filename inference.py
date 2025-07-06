import torchvision.transforms as T
from PIL import Image
import argparse
from network.model import get_model
import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference for Multi-Label Classification')

    parser.add_argument('--model_name', default="EfficientNetB3Pretrained")
    parser.add_argument('--model_path', default="./docs/logs/Mod.pth")
    parser.add_argument('--device', default="cuda")
    parser.add_argument('--image_path', default="data/preprocessed_images/0_left.jpg")
    args = parser.parse_args()

    params = vars(args)
    print(params)

    # ✅ Device check (auto fallback to CPU if CUDA is unavailable)
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Using device: {device}")

    # ✅ Load model
    model = get_model(args.model_name, device, {})
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    # ✅ Image transformations
    img_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # ✅ Class labels
    classes = {
        0: "N",  # Normal
        1: "D",  # Diabetic retinopathy
        2: "G",  # Glaucoma
        3: "C",  # Cataract
        4: "A",  # Age-related Macular Degeneration
        5: "H",  # Hypertension
        6: "M",  # Myopia
        7: "O"   # Other diseases
    }

    # ✅ Load and preprocess the image
    img = Image.open(args.image_path).convert('RGB')
    img = img_transform(img).unsqueeze(0).to(device)

    # ✅ Forward pass
    with torch.no_grad():
        output = model(img)
        probs = torch.sigmoid(output)

    # ✅ Apply threshold (for multi-label prediction)
    threshold = 0.5
    preds = (probs > threshold).cpu().numpy().astype(int).flatten()

    # ✅ Map predictions to class names
    predicted_classes = [classes[i] for i, val in enumerate(preds) if val == 1]

    if predicted_classes:
        print("Predicted Classes:", predicted_classes)
    else:
        print("No disease detected (or below threshold)")
