import torch
import os
import pandas as pd
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
import argparse

# Define the transform for input images
transform = transforms.Compose([
    transforms.Resize([240, 240]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_models(base_model_paths, meta_model_path, device='cuda'):
    """
    Load the base models and meta model from the saved paths.
    Args:
        base_model_paths (list): List of file paths to the saved base models.
        meta_model_path (str): File path to the saved meta model.
        device (str): Device to load the models onto, 'cuda' or 'cpu'.
    Returns:
        base_models (list): List of loaded base models.
        meta_model (nn.Module): Loaded meta model.
    """
    base_models = []
    
    # Load each base model's state_dict and load it into the model
    for model_path in base_model_paths:
        model = torch.load(model_path, map_location=device)
        model = model.to(device)  # Move model to device
        base_models.append(model)

    # Load meta model
    meta_model = torch.load(meta_model_path, map_location=device)
    meta_model = meta_model.to(device)

    return base_models, meta_model

def run_inference(image_dir, base_models, meta_model, device='cuda'):
    """
    Run inference on the images in the given directory using the trained models.
    Args:
        image_dir (str): Directory containing images.
        base_models (list): List of base models.
        meta_model (nn.Module): The meta model for final decision.
        device (str): Device to run the inference on, 'cuda' or 'cpu'.
    Returns:
        predictions (list): List of tuples with image names and final decision predictions.
    """
    predictions = []

    # Process each image in the directory
    for image_name in tqdm(os.listdir(image_dir), desc="Processing Images"):
        image_path = os.path.join(image_dir, image_name)
        if not image_path.lower().endswith(('png', 'jpg', 'jpeg')):
            continue

        # Open and transform image
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Collect base model outputs
        base_outputs = []
        for model in base_models:
            with torch.no_grad():
                output = model(image_tensor).squeeze().cpu()  # Get output and move to CPU
                
                # Ensure output is at least 2D (for concatenation)
                if output.ndimension() == 1:  # If output is 1D, unsqueeze it to make it 2D
                    output = output.unsqueeze(0)
                elif output.ndimension() == 0:  # If output is scalar, unsqueeze twice
                    output = output.unsqueeze(0).unsqueeze(0)
                
                base_outputs.append(output)

        # Ensure all base model outputs are concatenated along dimension 1
        base_outputs = torch.cat(base_outputs, dim=1).to(device)

        # Final prediction using meta model
        with torch.no_grad():
            final_decision = meta_model(image_tensor, base_outputs).squeeze(-1).cpu()

        # Convert to binary output (0 or 1)
        final_decision = torch.sigmoid(final_decision) > 0.5  # Thresholding
        final_decision = int(final_decision.item())

        predictions.append((image_name, final_decision))

    return predictions

def save_predictions(predictions, output_csv):
    """
    Save the predictions to a CSV file.
    Args:
        predictions (list): List of tuples containing image name and final decision.
        output_csv (str): Path to save the output CSV.
    """
    df = pd.DataFrame(predictions, columns=["Image Name", "Final Decision"])
    df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run inference on images using trained models.")
    
    # Add argument for input image directory
    parser.add_argument('--input', type=str, required=True, help="Directory containing images for inference.")
    
    # Add argument for base models
    parser.add_argument('--basemodels', nargs='+', required=True, help="Paths to the base model files.")
    
    # Add argument for meta model
    parser.add_argument('--metamodel', type=str, required=True, help="Path to the meta model file.")
    
    # Add argument for output CSV file
    parser.add_argument('--output', type=str, required=True, help="Path to save the output CSV file.")
    
    # Add argument for device (cuda or cpu)
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help="Device to run the inference on.")
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()

    # Load models
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    base_models, meta_model = load_models(args.basemodels, args.metamodel, device)

    # Run inference
    predictions = run_inference(args.input, base_models, meta_model, device)

    # Save predictions to CSV
    save_predictions(predictions, args.output)
