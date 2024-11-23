import cv2
import numpy as np
from PIL import Image
from skimage.color import rgb2lab
from sklearn.cluster import KMeans


def initialize_coords_from_image_using_segmentation(image_path, number_of_samples):
    """
    Initialize coordinates by segmenting an image and placing points at segment centroids.

    Args:
        image_path (str): Path to the input image.
        number_of_samples (int): Number of initial coordinates to generate.

    Returns:
        torch.Tensor: Tensor of shape (number_of_samples, 2) with initial coordinates.
    """
    # Load the image with PIL and convert to RGB
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    # Convert the image to a NumPy array and reshape it for clustering
    image_np = np.array(image)
    h, w, _ = image_np.shape
    reshaped_image = rgb2lab(image_np).reshape(
        (-1, 3)
    )  # Convert to LAB color space for clustering

    # Perform k-means clustering to segment the image
    kmeans = KMeans(n_clusters=number_of_samples,
                    random_state=0).fit(reshaped_image)
    labels = kmeans.labels_.reshape((h, w))

    # Calculate the centroid for each segment
    initial_coords = []
    for i in range(number_of_samples):
        # Get coordinates of pixels in the current cluster
        yx_coords = np.column_stack(np.where(labels == i))

        # Calculate centroid if the cluster is not empty
        if yx_coords.size > 0:
            centroid = yx_coords.mean(axis=0)
            initial_coords.append(centroid)

    # Convert centroids to integer coordinates and clamp within bounds
    initial_coords = np.clip(
        np.array(initial_coords).astype(int), [0, 0], [width - 1, height - 1]
    )

    return initial_coords


def initialize_coords_from_image_using_random(image_array, number_of_samples):
    width, height, _ = image_array.shape
    initial_coords = np.random.randint(
        0, [width, height], size=(number_of_samples, 2))

    return initial_coords


def initialize_coords_using_edges(image_path):
    """
    Initialize coordinates by detecting edges in the image.
    Returns all edge points found without forcing a specific number.

    Args:
        image_path (str): Path to the input image

    Returns:
        np.ndarray: Array of shape (N, 2) with coordinates in [x, y] format
    """
    # Read image and convert to grayscale
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    # Detect edges using Canny
    edges = cv2.Canny(gray, 50, 200)

    # Get coordinates of edge pixels (returns in y,x format)
    edge_coords = np.column_stack(np.where(edges > 0))

    # Convert to x,y format
    return np.column_stack((edge_coords[:, 1], edge_coords[:, 0]))


def initialize_coords_using_sift(image_path, contrast_threshold=0.01):
    """
    Initialize coordinates using SIFT feature detection.
    Returns all features found without forcing a specific number.

    Args:
        image_path (str): Path to the input image
        contrast_threshold (float): Contrast threshold for SIFT detection

    Returns:
        np.ndarray: Array of shape (N, 2) with coordinates in [x, y] format
    """
    # Read image
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    # Initialize SIFT detector with parameters that affect number of features
    sift = cv2.SIFT_create(
        contrastThreshold=contrast_threshold,  # Adjust this to get more/fewer points
        edgeThreshold=10,
        nOctaveLayers=3,
    )

    # Detect keypoints
    keypoints = sift.detect(gray, None)

    if len(keypoints) > 0:
        # Extract coordinates and round to integers
        coords = np.round([[kp.pt[0], kp.pt[1]]
                          for kp in keypoints]).astype(int)

        # Ensure coordinates are within image bounds
        height, width = gray.shape
        coords[:, 0] = np.clip(coords[:, 0], 0, width - 1)
        coords[:, 1] = np.clip(coords[:, 1], 0, height - 1)

        # Remove any duplicate coordinates that
        return coords
    else:
        return np.array([], dtype=int)
