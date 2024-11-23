import torch.nn.functional as F
import torch.nn as nn
import torch
from helpers import create_gaussian_window


def ssim(
    img_source: torch.Tensor, img_target: torch.Tensor, window_size: int = 11
) -> torch.Tensor:
    """
    Calculate the Structural Similarity Index (SSIM) between two images.
    A lower value indicates more similarity between the images.

    Args:
        img_source (torch.Tensor): Source image tensor of shape (H, W, C)
        img_target (torch.Tensor): Target image tensor of shape (H, W, C)
        window_size (int, optional): Size of the gaussian kernel. Defaults to 11.

    Returns:
        torch.Tensor: SSIM loss between the images, clamped to [0, 1]
    """
    # Get number of channels from input image
    (_, _, num_channels) = img_source.size()

    # Reshape tensors to (B, C, H, W) format required by conv2d
    img_source = img_source.unsqueeze(0).permute(0, 3, 1, 2)
    img_target = img_target.unsqueeze(0).permute(0, 3, 1, 2)

    # Constants for numerical stability
    STABILITY_CONST_1 = 0.01**2  # C1 in the SSIM formula
    STABILITY_CONST_2 = 0.03**2  # C2 in the SSIM formula

    # Create gaussian window for filtering
    gaussian_window = create_gaussian_window(window_size, num_channels)
    gaussian_window = gaussian_window.type_as(img_source)

    # Calculate mean of both images using gaussian window
    mean_source = F.conv2d(
        img_source, gaussian_window, padding=window_size // 2, groups=num_channels
    )
    mean_target = F.conv2d(
        img_target, gaussian_window, padding=window_size // 2, groups=num_channels
    )

    # Calculate squared means
    mean_source_sq = mean_source.pow(2)
    mean_target_sq = mean_target.pow(2)
    mean_product = mean_source * mean_target

    # Calculate variances and covariance using gaussian window
    variance_source = (
        F.conv2d(
            img_source * img_source,
            gaussian_window,
            padding=window_size // 2,
            groups=num_channels,
        )
        - mean_source_sq
    )
    variance_target = (
        F.conv2d(
            img_target * img_target,
            gaussian_window,
            padding=window_size // 2,
            groups=num_channels,
        )
        - mean_target_sq
    )
    covariance = (
        F.conv2d(
            img_source * img_target,
            gaussian_window,
            padding=window_size // 2,
            groups=num_channels,
        )
        - mean_product
    )

    # Calculate SSIM
    numerator = (2 * mean_product + STABILITY_CONST_1) * (
        2 * covariance + STABILITY_CONST_2
    )
    denominator = (mean_source_sq + mean_target_sq + STABILITY_CONST_1) * (
        variance_source + variance_target + STABILITY_CONST_2
    )
    ssim_map = numerator / denominator

    # Convert SSIM to a distance metric and clamp to [0, 1]
    return torch.clamp((1 - ssim_map) / 2, 0, 1)


def d_ssim_loss(
    img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11
) -> torch.Tensor:
    """
    Calculate the mean SSIM distance between two images.

    Args:
        img1 (torch.Tensor): First image tensor
        img2 (torch.Tensor): Second image tensor
        window_size (int, optional): Size of gaussian kernel. Defaults to 11.

    Returns:
        torch.Tensor: Mean SSIM distance
    """
    return ssim(img1, img2, window_size).mean()


def combined_loss(
    pred: torch.Tensor, target: torch.Tensor, lambda_param: float = 0.2
) -> torch.Tensor:
    """
    Combine L1 loss and SSIM loss with a weighting parameter.

    Args:
        pred (torch.Tensor): Predicted image tensor
        target (torch.Tensor): Target image tensor
        lambda_param (float, optional): Weight for combining losses. Defaults to 0.2.
            lambda_param controls the balance between L1 and SSIM loss:
            - Higher values give more weight to SSIM loss
            - Lower values give more weight to L1 loss

    Returns:
        torch.Tensor: Combined weighted loss
    """
    l1_criterion = nn.L1Loss()
    l1_component = (1 - lambda_param) * l1_criterion(pred, target)
    ssim_component = lambda_param * d_ssim_loss(pred, target)
    return l1_component + ssim_component
