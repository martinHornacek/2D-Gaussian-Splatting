import numpy as np
import torch.nn.functional as F
import torch


def generate_2D_gaussian_splatting(
    kernel_size: int,
    scale: torch.Tensor,
    rotation: torch.Tensor,
    coords: torch.Tensor,
    colours: torch.Tensor,
    image_size: tuple = (256, 256, 3),
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a 2D image by splatting multiple Gaussian kernels with different properties.

    Args:
        kernel_size (int): Size of each Gaussian kernel
        scale (torch.Tensor): Scale factors for x and y directions, shape (N, 2)
        rotation (torch.Tensor): Rotation angles in radians, shape (N,)
        coords (torch.Tensor): Center coordinates for each Gaussian, shape (N, 2)
        colours (torch.Tensor): RGB colors for each Gaussian, shape (N, 3)
        image_size (tuple): Output image dimensions (H, W, C), default (256, 256, 3)
        device (str): Computation device, default "cpu"

    Returns:
        torch.Tensor: Generated image of shape (H, W, 3)

    Raises:
        ValueError: If kernel_size is larger than image dimensions
    """
    # Get number of Gaussians from input
    num_gaussians = colours.shape[0]

    # Reshape scale and rotation tensors to correct dimensions
    scale = scale.view(num_gaussians, 2)
    rotation = rotation.view(num_gaussians)

    # Calculate rotation matrix components
    cos_theta = torch.cos(rotation)
    sin_theta = torch.sin(rotation)

    # Construct rotation matrices for all Gaussians
    rotation_matrices = torch.stack(
        [
            torch.stack([cos_theta, -sin_theta], dim=-1),
            torch.stack([sin_theta, cos_theta], dim=-1),
        ],
        dim=-2,
    )

    # Create diagonal scale matrices
    scale_matrices = torch.diag_embed(scale)

    # Compute covariance matrices for all Gaussians: R * S * S * R^T
    covariance_matrices = (
        rotation_matrices
        @ scale_matrices
        @ scale_matrices
        @ rotation_matrices.transpose(-1, -2)
    )

    # Calculate inverse of covariance matrices
    inv_covariance_matrices = torch.inverse(covariance_matrices)

    # Create coordinate grid for kernel
    grid_coords = torch.linspace(-5, 5, kernel_size, device=device)
    grid_x, grid_y = torch.meshgrid(grid_coords, grid_coords, indexing="ij")
    grid_points = (
        torch.stack([grid_x, grid_y], dim=-1)
        .unsqueeze(0)
        .expand(num_gaussians, -1, -1, -1)
    )

    # Compute Gaussian values using matrix operations
    exponent = torch.einsum(
        "bxyi,bij,bxyj->bxy", grid_points, -0.5 * inv_covariance_matrices, grid_points
    )
    normalization_factor = (
        2
        * torch.tensor(np.pi, device=device)
        * torch.sqrt(torch.det(covariance_matrices))
    )
    kernel = torch.exp(exponent) / \
        normalization_factor.view(num_gaussians, 1, 1)

    # Normalize kernels to [0, 1] range
    kernel_peaks = kernel.amax(dim=(-2, -1), keepdim=True)
    normalized_kernel = kernel / kernel_peaks

    # Expand kernel for RGB channels
    rgb_kernel = normalized_kernel.unsqueeze(1).expand(-1, 3, -1, -1)

    # Calculate padding to match target image size
    pad_height = image_size[0] - kernel_size
    pad_width = image_size[1] - kernel_size

    if pad_height < 0 or pad_width < 0:
        raise ValueError(
            "Kernel size must be smaller than or equal to the image dimensions"
        )

    # Add padding to kernels
    padding_sizes = (
        pad_width // 2,
        pad_width // 2 + pad_width % 2,
        pad_height // 2,
        pad_height // 2 + pad_height % 2,
    )
    padded_rgb_kernel = F.pad(
        rgb_kernel, padding_sizes, mode="constant", value=0)

    # Prepare transformation matrices for translation
    batch_size, channels, height, width = padded_rgb_kernel.shape
    translation_matrices = torch.zeros(
        batch_size, 2, 3, dtype=torch.float32, device=device
    )
    translation_matrices[:, 0, 0] = 1.0  # Scale x
    translation_matrices[:, 1, 1] = 1.0  # Scale y
    translation_matrices[:, :, 2] = coords  # Translation

    # Apply translations using grid sampling
    sampling_grid = F.affine_grid(
        translation_matrices,
        size=(batch_size, channels, height, width),
        align_corners=True,
    )
    translated_rgb_kernel = F.grid_sample(
        padded_rgb_kernel, sampling_grid, align_corners=True
    )

    # Apply colors to kernels and combine
    rgb_colours = colours.unsqueeze(-1).unsqueeze(-1)
    colored_gaussian_layers = rgb_colours * translated_rgb_kernel
    combined_image = colored_gaussian_layers.sum(dim=0)

    # Post-process final image
    final_image = torch.clamp(combined_image, 0, 1)
    final_image = final_image.permute(1, 2, 0)  # Convert to HWC format

    return final_image


def get_normalized_coords_and_colors(
    image_array, input_coords, image_size, device: str = "cpu"
):
    """
    Convert pixel coordinates to normalized coordinates and extract corresponding color values.

    Args:
        image_array: Image as numpy array
        input_coords: Pixel coordinates in [x, y] format
        image_size: Image dimensions as (height, width)
        device (str): Computation device, default "cpu"

    Returns:
        Tuple containing color values and normalized coordinates
    """
    # Normalize pixel coordinates to [0,1] range by dividing by image dimensions
    normalized_coords = torch.tensor(
        input_coords / [image_size[0], image_size[1]], device=device
    ).float()

    # Define the center point in normalized space
    center_point = torch.tensor([0.5, 0.5], device=device).float()

    # Convert coordinates to [-1,1] range and center them
    normalized_coords = (center_point - normalized_coords) * 2.0

    # Extract RGB values for each coordinate from the image
    pixel_colors = [image_array[coord[1], coord[0]] for coord in input_coords]
    pixel_colors_np = np.array(pixel_colors)
    pixel_colors_tensor = torch.tensor(pixel_colors_np, device=device).float()

    return pixel_colors_tensor, normalized_coords


def prune_low_opacity_gaussians(W, active_gaussians_mask):
    """
    Remove Gaussian points with opacity values below threshold.

    Args:
        W: Parameter tensor containing Gaussian properties
        active_gaussians_mask: Boolean mask of active Gaussians
    """
    indices_to_remove = (torch.sigmoid(
        W[:, 3]) < 0.01).nonzero(as_tuple=True)[0]

    if len(indices_to_remove) > 0:
        print(f"Number of pruned points: {len(indices_to_remove)}")
        active_gaussians_mask[indices_to_remove] = False

    # Zero-out parameters for inactive Gaussians
    W.data[~active_gaussians_mask] = 0.0


def process_gaussian_batch(output, image_size, kernel_size, device):
    """
    Process a batch of Gaussian parameters into renderable properties.

    Args:
        output: Tensor of Gaussian parameters
        image_size: Target image dimensions
        device: Computation device

    Returns:
        Tuple of processed Gaussian properties
    """
    # Convert raw parameters to usable properties
    scale = torch.sigmoid(output[:, 0:2])
    rotation = np.pi / 2 * torch.tanh(output[:, 2])
    alpha = torch.sigmoid(output[:, 3])
    colours = torch.sigmoid(output[:, 4:7])
    pixel_coords = torch.tanh(output[:, 7:9])

    batch_size = output.shape[0]
    colours_with_alpha = colours * alpha.view(batch_size, 1)

    g_tensor_batch = generate_2D_gaussian_splatting(
        kernel_size, scale, rotation, pixel_coords, colours_with_alpha, image_size, device=device
    )

    return g_tensor_batch


def perform_densification(
    W,
    active_gaussians_mask,
    current_sample_count,
    gradient_threshold,
    gaussian_threshold,
):
    """
    Perform densification by splitting and cloning Gaussians based on gradients.

    Args:
        W: Parameter tensor
        active_gaussians_mask: Boolean mask of active Gaussians
        current_sample_count: Current number of active Gaussians
        gradient_threshold: Threshold for considering gradients as large
        gaussian_threshold: Threshold for considering Gaussian scale as large

    Returns:
        Updated current_sample_count
    """
    # Calculate gradient and Gaussian norms
    gradient_norms = torch.norm(
        W.grad[active_gaussians_mask][:, 7:9], dim=1, p=2)
    gaussian_norms = torch.norm(
        torch.sigmoid(W.data[active_gaussians_mask][:, 0:2]), dim=1, p=2
    )

    # Sort gradients and Gaussian scales
    sorted_grads, sorted_grads_indices = torch.sort(
        gradient_norms, descending=True)
    sorted_gauss, sorted_gauss_indices = torch.sort(
        gaussian_norms, descending=True)

    # Find points exceeding thresholds
    large_gradient_mask = sorted_grads > gradient_threshold
    large_gradient_indices = sorted_grads_indices[large_gradient_mask]

    large_gauss_mask = sorted_gauss > gaussian_threshold
    large_gauss_indices = sorted_gauss_indices[large_gauss_mask]

    # Find common points (large gradient AND large scale)
    common_indices_mask = torch.isin(
        large_gradient_indices, large_gauss_indices)
    common_indices = large_gradient_indices[common_indices_mask]
    distinct_indices = large_gradient_indices[~common_indices_mask]

    # Split points with both large gradients and scales
    if len(common_indices) > 0:
        print(f"Number of splitted points: {len(common_indices)}")
        current_sample_count = split_gaussians(
            W, active_gaussians_mask, common_indices, current_sample_count
        )

    # Clone points with large gradients but small scales
    if len(distinct_indices) > 0:
        print(f"Number of cloned points: {len(distinct_indices)}")
        current_sample_count = clone_and_move_gaussians(
            W, active_gaussians_mask, distinct_indices, current_sample_count
        )

    return current_sample_count


def split_gaussians(W, active_gaussians_mask, indices, current_sample_count):
    """
    Split Gaussians by creating copies with reduced scale.
    """
    start_index = current_sample_count + 1
    end_index = current_sample_count + 1 + len(indices)
    active_gaussians_mask[start_index:end_index] = True
    W.data[start_index:end_index, :] = W.data[indices, :]

    scale_reduction_factor = 1.6
    W.data[start_index:end_index, 0:2] /= scale_reduction_factor
    W.data[indices, 0:2] /= scale_reduction_factor

    return current_sample_count + len(indices)


def clone_and_move_gaussians(W, active_gaussians_mask, indices, current_sample_count):
    """
    Clone Gaussians and move them in the gradient direction.
    """
    start_index = current_sample_count + 1
    end_index = current_sample_count + 1 + len(indices)
    active_gaussians_mask[start_index:end_index] = True
    W.data[start_index:end_index, :] = W.data[indices, :]

    # Calculate movement direction
    positional_gradients = W.grad[indices, 7:9]
    gradient_magnitudes = torch.norm(positional_gradients, dim=1, keepdim=True)
    normalized_gradients = positional_gradients / (gradient_magnitudes + 1e-8)

    # Move cloned Gaussians
    step_size = 0.01
    W.data[start_index:end_index, 7:9] += step_size * normalized_gradients

    return current_sample_count + len(indices)
