from network.model import get_model
import torch
from tqdm import tqdm
import argparse
import time


def fps_test(args):
    # Check device availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA is not available. Switching to CPU.")
        args.device = "cpu"

    device = torch.device(args.device)

    # Create dummy input
    inp = torch.randn(1, 3, 224, 224).to(device)

    # Load model
    model = get_model(args.model_name, args.device, {})
    model.eval().to(device)

    print(f"\nTesting FPS for model: {args.model_name} on device: {args.device}")
    print("=" * 60)

    time_list = []

    # Warm-up to stabilize GPU clocks or CPU cache
    for _ in range(args.warmup):
        _ = model(inp)

    if args.device == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        for _ in tqdm(range(args.iter_count), desc="Running Inference"):
            start.record()
            _ = model(inp)
            end.record()

            torch.cuda.synchronize()
            sec = start.elapsed_time(end) / 1000  # Convert ms to seconds
            time_list.append(sec)

    else:  # CPU timing
        for _ in tqdm(range(args.iter_count), desc="Running Inference"):
            start = time.time()
            _ = model(inp)
            end = time.time()

            sec = end - start
            time_list.append(sec)

    avg_time = sum(time_list) / len(time_list)
    avg_fps = 1 / avg_time if avg_time > 0 else float('inf')

    print(f"\nResults:")
    print(f"Model Name : {args.model_name}")
    print(f"Device     : {args.device}")
    print(f"Avg Time per Inference: {avg_time:.6f} seconds")
    print(f"Average FPS           : {avg_fps:.2f} frames per second")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FPS Tester for Models")

    parser.add_argument('--model_name', default="MobileNetV2Pretrained", type=str, help='Model name')
    parser.add_argument('--warmup', default=20, type=int, help='Warm-up iterations (excluded from FPS)')
    parser.add_argument('--iter_count', default=100, type=int, help='Number of iterations for FPS test')
    parser.add_argument('--device', default="cuda", type=str, choices=["cuda", "cpu"], help='Device: cuda or cpu')

    args = parser.parse_args()

    print(args)

    fps_test(args)
