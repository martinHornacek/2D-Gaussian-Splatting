import cv2
import numpy as np
from skimage.color import rgb2lab
from sklearn.cluster import KMeans


def initialize_coords_from_image_using_segmentation(image_array, number_of_samples):
    """
    Initialize coordinates by segmenting an image and placing points at segment centroids.

    Args:
        image_array (np.ndarray): Input image as numpy array of shape (width, height, channels)
                                 with float64 values in range [0.0, 1.0]
        number_of_samples (int): Number of initial coordinates to generate.

    Returns:
        tuple: (np.ndarray, np.ndarray, np.ndarray) containing:
            - Array of shape (number_of_samples, 2) containing [x, y] coordinates
            - Labels array of shape (height, width) containing segment labels
            - Original uint8 image array
    """
    # Convert float64 [0.0, 1.0] to uint8 [0, 255]
    image_array_uint8 = (image_array * 255).astype(np.uint8)

    # Get image dimensions from the array shape
    height, width, _ = image_array_uint8.shape

    # Convert to LAB color space for clustering
    lab_image = rgb2lab(image_array_uint8).reshape((-1, 3))

    # Perform k-means clustering to segment the image
    kmeans = KMeans(n_clusters=number_of_samples,
                    random_state=0).fit(lab_image)
    labels = kmeans.labels_.reshape((height, width))

    # Calculate the centroid for each segment
    initial_coords = []
    for i in range(number_of_samples):
        # Get coordinates of pixels in the current cluster
        y_coords, x_coords = np.where(labels == i)

        # Calculate centroid if the cluster is not empty
        if len(x_coords) > 0:
            centroid_x = int(np.mean(x_coords))
            centroid_y = int(np.mean(y_coords))
            initial_coords.append([centroid_x, centroid_y])

    # Convert to numpy array and ensure we have the right number of coordinates
    initial_coords = np.array(initial_coords)

    # If we have fewer coordinates than requested (due to empty clusters),
    # fill in with random coordinates
    if len(initial_coords) < number_of_samples:
        remaining_samples = number_of_samples - len(initial_coords)
        random_coords = np.random.randint(
            0, [width, height],
            size=(remaining_samples, 2)
        )
        initial_coords = np.vstack([initial_coords, random_coords])

    # Ensure coordinates are within bounds
    initial_coords[:, 0] = np.clip(initial_coords[:, 0], 0, width - 1)
    initial_coords[:, 1] = np.clip(initial_coords[:, 1], 0, height - 1)

    return initial_coords, labels, image_array_uint8


def initialize_coords_from_image_using_random(image_array, number_of_samples):
    """
    Initialize coordinates randomly within the image dimensions.

    Args:
        image_array (np.ndarray): Input image as a numpy array of shape (width, height, channels)
        number_of_samples (int): Number of coordinate pairs to generate

    Returns:
        np.ndarray: Array of shape (number_of_samples, 2) containing [x, y] coordinates
    """
    # Get image dimensions from the array shape
    width, height, _ = image_array.shape

    # Generate random coordinates within image boundaries
    # np.random.randint generates integers from [0, width) and [0, height)
    initial_coords = np.random.randint(
        0, [width, height], size=(number_of_samples, 2))

    return initial_coords


def initialize_coords_from_image_using_edges(
    image_array,
    threshold1=100,
    threshold2=150,
    max_points=5000,
    min_distance=7
):
    """
    Find coordinates of edge pixels in the image using Canny edge detection,
    with controls for the number of points returned.

    Args:
        image_array (np.ndarray): Input image as numpy array of shape (width, height, channels) 
                                 with float64 values in range [0.0, 1.0]
        threshold1 (int): First threshold for Canny edge detector (default: 100)
        threshold2 (int): Second threshold for Canny edge detector (default: 150)
        max_points (int): Maximum number of points to return (default: 1000)
        min_distance (int): Minimum pixel distance between points (default: 10)

    Returns:
        np.ndarray: Array of shape (N, 2) containing [x, y] coordinates of selected edge pixels,
                   where N is the reduced number of edge pixels
    """
    # Convert float64 [0.0, 1.0] to uint8 [0, 255]
    image_array_uint8 = (image_array * 255).astype(np.uint8)

    # Convert to grayscale without transposing
    if len(image_array_uint8.shape) == 3:
        gray_image = cv2.cvtColor(image_array_uint8, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image_array_uint8

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, threshold1, threshold2)

    # Get coordinates of all edge pixels
    edge_x, edge_y = np.where(edges > 0)
    edge_coords = np.column_stack((edge_y, edge_x))

    # If we have more points than desired, reduce them
    if len(edge_coords) > max_points:
        # Method 1: Non-maximal suppression using distance
        selected_indices = []
        remaining_indices = set(range(len(edge_coords)))

        while len(selected_indices) < max_points and remaining_indices:
            # Pick a random index from remaining points
            idx = np.random.choice(list(remaining_indices))
            selected_indices.append(idx)

            # Remove nearby points
            current_point = edge_coords[idx]
            indices_to_remove = set()

            for other_idx in remaining_indices:
                if other_idx == idx:
                    continue

                other_point = edge_coords[other_idx]
                distance = np.sqrt(np.sum((current_point - other_point) ** 2))

                if distance < min_distance:
                    indices_to_remove.add(other_idx)

            remaining_indices -= indices_to_remove
            remaining_indices.remove(idx)

            if not remaining_indices:
                break

        edge_coords = edge_coords[selected_indices]

        # If we still have too many points, randomly sample
        if len(edge_coords) > max_points:
            indices = np.random.choice(
                len(edge_coords),
                size=max_points,
                replace=False
            )
            edge_coords = edge_coords[indices]

    return edge_coords


def initialize_coords_using_sift(image_array, contrast_threshold=0.001):
    """
    Initialize coordinates using SIFT feature detection.
    Returns all features found without forcing a specific number.

    Args:
        image_input (str or np.ndarray): Path to the input image or a grayscale image array.
        contrast_threshold (float): Contrast threshold for SIFT detection.

    Returns:
        np.ndarray: Array of shape (N, 2) with coordinates in [x, y] format.
    """
    # Convert float64 [0.0, 1.0] to uint8 [0, 255]
    image_array_uint8 = (image_array * 255).astype(np.uint8)

    # Convert to grayscale without transposing
    if len(image_array_uint8.shape) == 3:
        gray_image = cv2.cvtColor(image_array_uint8, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image_array_uint8

    # Initialize SIFT detector
    sift = cv2.SIFT_create(
        contrastThreshold=contrast_threshold,
        edgeThreshold=5,
        nOctaveLayers=3,
    )

    # Detect keypoints
    keypoints = sift.detect(gray_image, None)

    if len(keypoints) > 0:
        # Extract coordinates and round to integers
        coords = np.round([[kp.pt[0], kp.pt[1]]
                          for kp in keypoints]).astype(int)

        # Ensure coordinates are within image bounds
        height, width = gray_image.shape
        coords[:, 0] = np.clip(coords[:, 0], 0, width - 1)
        coords[:, 1] = np.clip(coords[:, 1], 0, height - 1)

        # Remove duplicates if necessary
        coords = np.unique(coords, axis=0)

        return coords
    else:
        return np.array([], dtype=int)
