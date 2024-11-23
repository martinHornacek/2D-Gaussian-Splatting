import numpy as np
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt


def get_normalized_gaussian(window_size: int, sigma: float) -> torch.Tensor:
    """
    Generate a 1D Gaussian distribution.

    Args:
        window_size (int): Size of the gaussian kernel
        sigma (float): Standard deviation of the gaussian distribution

    Returns:
        torch.Tensor: Normalized 1D gaussian distribution
    """
    # Calculate gaussian values for each position in the window
    gaussian_values = torch.exp(
        torch.tensor(
            [
                -((position - window_size // 2) ** 2) / float(2 * sigma**2)
                for position in range(window_size)
            ]
        )
    )

    # Normalize the distribution
    normalized_gaussian = gaussian_values / gaussian_values.sum()
    return normalized_gaussian


def create_gaussian_window(
    window_size: int, num_channels: int
) -> torch.autograd.Variable:
    """
    Create a 2D Gaussian window for multiple channels.

    Args:
        window_size (int): Size of the gaussian kernel
        num_channels (int): Number of channels in the input

    Returns:
        torch.autograd.Variable: 2D gaussian window expanded for all channels
    """
    # Create 1D gaussian window
    gaussian_1d = get_normalized_gaussian(window_size, sigma=1.5).unsqueeze(1)

    # Create 2D gaussian window by multiplying 1D gaussian with its transpose
    gaussian_2d = gaussian_1d.mm(gaussian_1d.t())
    gaussian_2d = gaussian_2d.float().unsqueeze(0).unsqueeze(0)

    # Expand the 2D window to all channels and create Variable
    multi_channel_window = torch.autograd.Variable(
        gaussian_2d.expand(num_channels, 1, window_size,
                           window_size).contiguous()
    )

    return multi_channel_window


def save_visualization(g_tensor_batch, epoch, directory, num_epochs, output, loss):
    """
    Save the current state as an image.
    """
    generated_array = g_tensor_batch.cpu().detach().numpy()
    img = Image.fromarray((generated_array * 255).astype(np.uint8))
    filename = f"{epoch}.jpg"
    file_path = os.path.join(directory, filename)
    img.save(file_path)
    print(
        f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, on {len(output)} points."
    )


def visualize_initial_points(image_array, initial_coords):
    """
    Visualize the initial points overlaid on the input image.

    Args:
        image_array (np.ndarray): Image as numpy array
        initial_coords (np.ndarray): Array of shape (N, 2) containing [x, y] coordinates
        title (str): Title for the plot
        save_path (str, optional): If provided, saves the visualization to this path
    """
    # Read and display the image
    plt.figure(figsize=(12, 8))

    # Display image
    plt.imshow(image_array)

    # Plot points over the image
    plt.scatter(
        initial_coords[:, 0],  # x coordinates
        initial_coords[:, 1],  # y coordinates
        c="red",  # point color
        s=10,  # point size
        alpha=0.6,  # transparency
        label=f"Initial Points (n={len(initial_coords)})",
    )

    plt.axis("off")  # Hide axes

    # Add grid for better point position visibility
    plt.grid(False)

    # Tight layout to prevent cutting off elements
    plt.tight_layout()
    plt.show()
    plt.close()
