import torch
import argparse
import os
import glob
import matplotlib.pyplot as plt

from gan import GAN


def load_model(model_path, device="cpu"):
    """Load a saved GAN model"""
    gan = GAN(0, 0)
    gan.load_state_dict(torch.load(model_path, map_location=device))

    return gan


def generate_samples(gan, num_samples=25, device="cpu"):
    """Generate samples using the trained generator"""
    gan.generator.eval()
    with torch.no_grad():
        noise = torch.randn(num_samples, gan.gen_input_dim, device=device)
        generated_samples = gan.generator(noise)

        generated_samples = generated_samples.view(num_samples, 28, 28).cpu().numpy()

        fig, axes = plt.subplots(5, 5, figsize=(10, 10))
        for i, ax in enumerate(axes.flat):
            ax.imshow(generated_samples[i], cmap="gray")
            ax.axis("off")
        plt.tight_layout()
        return fig


def inference_single_model(model_path, output_dir="results", device="cpu"):
    """Perform inference on a single model"""
    print(f"Loading model: {model_path}")

    gan = load_model(model_path, device)
    model_name = os.path.basename(model_path).replace(".pkl", "")

    fig = generate_samples(gan, device=device)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{model_name}_samples.png")
    fig.savefig(output_path)
    plt.close(fig)

    print(f"Generated samples saved to: {output_path}")
    return output_path


def inference_all_models(models_dir="models", output_dir="results", device="cpu"):
    """Perform inference on all models in the models directory"""
    model_files = glob.glob(os.path.join(models_dir, "**", "*.pkl"), recursive=True)
    model_files.sort()

    if not model_files:
        print("No .pkl model files found in the models directory.")
        return

    print(f"Found {len(model_files)} models to process...")

    results = []
    for model_path in model_files:
        try:
            output_path = inference_single_model(model_path, output_dir, device)
            results.append(
                {
                    "model_path": model_path,
                    "output_path": output_path,
                    "status": "success",
                }
            )
        except Exception as e:
            print(f"Error processing {model_path}: {str(e)}")
            results.append(
                {
                    "model_path": model_path,
                    "output_path": None,
                    "status": f"error: {str(e)}",
                }
            )

    print(f"\nInference completed. Processed {len(results)} models.")
    print(f"Results saved in: {output_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(description="GAN Inference Script")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to specific model file for inference",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory containing model files (default: models)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save generated samples (default: results)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for inference (default: cpu)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run inference on all models in the models directory",
    )

    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    print(f"Using device: {device}")

    if args.model:
        if not os.path.exists(args.model):
            print(f"Model file not found: {args.model}")
            return

        inference_single_model(args.model, args.output_dir, device)

    elif args.batch:
        inference_all_models(args.models_dir, args.output_dir, device)

    else:
        print("Please specify either --model <path> for single model inference")
        print("or --batch to run inference on all models in the models directory")
        print("Use --help for more information.")


if __name__ == "__main__":
    main()
