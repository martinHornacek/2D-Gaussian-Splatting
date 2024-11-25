import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import os
import gc
import yaml
from datetime import datetime
from PIL import Image
from helpers import (
    save_visualization,
    visualize_initial_points,
)
from initialization_strategies import initialize_coords_from_image_using_random
from loss_calculation import combined_loss
from gaussian_splatting import (
    get_normalized_coords_and_colors,
    perform_densification,
    process_gaussian_batch,
    prune_low_opacity_gaussians,
)

torch.manual_seed(42)  # To achieve reproducibility of the results

# Load configuration settings from YAML file
with open("config.yml", "r") as config_file:
    config = yaml.safe_load(config_file)

# Extract model hyperparameters from config
KERNEL_SIZE = config["KERNEL_SIZE"]  # Size of Gaussian kernel
# Target dimensions for image processing
image_size = tuple(config["image_size"])
number_of_primary_samples = config[
    "primary_samples"
]  # Initial number of Gaussian samples
number_of_backup_samples = config[
    "backup_samples"
]  # Additional samples for densification
num_epochs = config["num_epochs"]  # Total training iterations
densification_interval = config[
    "densification_interval"
]  # Steps between adding new Gaussians
learning_rate = config["learning_rate"]  # Optimizer learning rate
input_image_path = config["image_file_name"]  # Path to source image
display_interval = config["display_interval"]  # Steps between visual updates
gradient_threshold = config[
    "gradient_threshold"
]  # Threshold for gradient-based densification
# Threshold for Gaussian importance
gaussian_threshold = config["gaussian_threshold"]

# Set up device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
total_number_of_gaussians = (
    number_of_primary_samples + number_of_backup_samples
)  # Total number of Gaussians

# Load and preprocess the input image
source_image = Image.open(input_image_path)
source_image = source_image.resize(
    (image_size[0], image_size[0])
)  # Resize to target dimensions
source_image = source_image.convert("RGB")  # Ensure RGB format
source_array = np.array(source_image)
normalized_array = source_array / 255.0  # Normalize pixel values to [0,1]
width, height, _ = normalized_array.shape

# Store processed image data
image_array = normalized_array
target_tensor = torch.tensor(image_array, dtype=torch.float32, device=device)

# Generate initial coordinates for Gaussians
initial_coords = initialize_coords_from_image_using_random(image_array, 343)
visualize_initial_points(image_array, initial_coords)

# The Total Number of Gaussian can change based on the output of the initialization algorithm
number_of_primary_samples, _ = initial_coords.shape

# Get normalized coordinates and corresponding colors
color_values, normalized_coords = get_normalized_coords_and_colors(
    image_array, initial_coords, image_size, total_number_of_gaussians
)

# Apply inverse sigmoid (logit) and inverse tanh transformations to ensure unconstrained optimization
color_values_transformed = torch.logit(
    color_values
)  # Transform colors to unconstrained space
coords_transformed = torch.atanh(
    normalized_coords
)  # Transform coordinates to unconstrained space

# Initialize Gaussian parameters
scale_params = torch.logit(
    torch.rand(total_number_of_gaussians, 2, device=device)
)  # Scale parameters for x,y
rotation_params = torch.atanh(
    2 * torch.rand(total_number_of_gaussians, 1, device=device) - 1
)  # Rotation angles
alpha_params = torch.logit(
    torch.rand(total_number_of_gaussians, 1, device=device)
)  # Transparency values

# Combine all parameters into single tensor
gaussian_params = torch.cat(
    [
        scale_params,
        rotation_params,
        alpha_params,
        color_values_transformed,
        coords_transformed,
    ],
    dim=1,
)

# Initialize masks for active and inactive Gaussians
active_samples = number_of_primary_samples
reserve_samples = total_number_of_gaussians - active_samples
active_gaussians_mask = torch.cat(
    [torch.ones(active_samples, dtype=bool),
     torch.zeros(reserve_samples, dtype=bool)],
    dim=0,
)
current_sample_count = active_samples

# Get current date and time as string
now = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

# Create a directory with the current date and time as its name
directory = f"{now}"
os.makedirs(directory, exist_ok=True)

W = nn.Parameter(gaussian_params)
optimizer = Adam([W], lr=learning_rate)
loss_history = []

"""
Training loop for Gaussian Splatting optimization with dynamic point management.
Includes pruning of low-opacity points and densification of high-gradient areas.
"""

# Main training loop
for epoch in range(num_epochs):
    # Memory management
    gc.collect()
    torch.cuda.empty_cache()

    # Prune low-opacity Gaussians periodically
    if epoch % densification_interval == 0 and epoch > 0:
        prune_low_opacity_gaussians(W, active_gaussians_mask)

    # Process active Gaussians
    output = W[active_gaussians_mask]
    g_tensor_batch = process_gaussian_batch(
        output, image_size, KERNEL_SIZE, device)

    # Calculate loss
    loss = combined_loss(g_tensor_batch, target_tensor, lambda_param=0.2)

    # Optimization step
    optimizer.zero_grad()
    loss.backward()

    # Zero gradients for inactive Gaussians
    if active_gaussians_mask is not None:
        W.grad.data[~active_gaussians_mask] = 0.0

    # Perform densification periodically
    if epoch % densification_interval == 0 and epoch > 0:
        current_sample_count = perform_densification(
            W,
            active_gaussians_mask,
            current_sample_count,
            gradient_threshold,
            gaussian_threshold,
        )

    optimizer.step()
    loss_history.append(loss.item())

    # Save visualization periodically
    if epoch % display_interval == 0:
        save_visualization(g_tensor_batch, epoch, directory,
                           num_epochs, output, loss)

final_loss = loss_history[-1]
print(
    f"Final Loss: {final_loss}."
)
