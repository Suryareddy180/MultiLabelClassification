import torchvision.transforms as T
from PIL import Image
import argparse
from network.model import get_model
import torch
import glob
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference for Multi-Label Classification')

    parser.add_argument('--model_name', default="EfficientNetB3Pretrained", help="Model architecture name")
    parser.add_argument('--log_dir', default="./docs/logs", help="Directory where models are saved")
    parser.add_argument('--device', default="cuda", help="cuda or cpu")
    parser.add_argument('--image_path', default="/kaggle/input/ocular-disease-recognition-odir5k/preprocessed_images/0_left.jpg", help="Path to input image")
    args = parser.parse_args()

    params = vars(args)
    print("Parameters:", params)

    # ✅ Device check
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Using device: {device}")

    # ✅ Auto-find latest model matching model_name
    model_files = sorted(glob.glob(os.path.join(args.log_dir, f"{args.model_name}_*.pth")))
    if not model_files:
        raise FileNotFoundError(f"No model found for name {args.model_name} in {args.log_dir}")
    latest_model = model_files[-1]
    print(f"✅ Loaded model: {latest_model}")

    # ✅ Load model
    model = get_model(args.model_name, device, {})
    model.load_state_dict(torch.load(latest_model, map_location=device))
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

    # ✅ Thresholding
    threshold = 0.5
    preds = (probs > threshold).cpu().numpy().astype(int).flatten()
    predicted_classes = [classes[i] for i, val in enumerate(preds) if val == 1]

    # ✅ Display Results
    if predicted_classes:
        print(f"Predicted Classes for {os.path.basename(args.image_path)}: {predicted_classes}")
    else:
        print("No disease detected (or below threshold)")
import torchvision.transforms as T
from PIL import Image
import argparse
from network.model import get_model
import torch
import glob
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference for Multi-Label Classification')

    parser.add_argument('--model_name', default="EfficientNetB3Pretrained", help="Model architecture name")
    parser.add_argument('--log_dir', default="./docs/logs", help="Directory where models are saved")
    parser.add_argument('--device', default="cuda", help="cuda or cpu")
    parser.add_argument('--image_path', default="/kaggle/input/ocular-disease-recognition-odir5k/preprocessed_images/0_left.jpg", help="Path to input image")
    args = parser.parse_args()

    params = vars(args)
    print("Parameters:", params)

    # ✅ Device check
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Using device: {device}")

    # ✅ Auto-find latest model matching model_name
    model_files = sorted(glob.glob(os.path.join(args.log_dir, f"{args.model_name}_*.pth")))
    if not model_files:
        raise FileNotFoundError(f"No model found for name {args.model_name} in {args.log_dir}")
    latest_model = model_files[-1]
    print(f"✅ Loaded model: {latest_model}")

    # ✅ Load model
    model = get_model(args.model_name, device, {})
    model.load_state_dict(torch.load(latest_model, map_location=device))
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

    # ✅ Thresholding
    threshold = 0.5
    preds = (probs > threshold).cpu().numpy().astype(int).flatten()
    predicted_classes = [classes[i] for i, val in enumerate(preds) if val == 1]

    # ✅ Display Results
    if predicted_classes:
        print(f"Predicted Classes for {os.path.basename(args.image_path)}: {predicted_classes}")
    else:
        print("No disease detected (or below threshold)")
