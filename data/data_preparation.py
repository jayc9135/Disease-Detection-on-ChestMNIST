import numpy as np
import sys
from torch.utils.data import Subset, DataLoader, TensorDataset
import torchvision.transforms as transforms


def _check_labels(dataset):
    """
    Check if all the labels are consistent or not
    :param dataset: The dataset to check
    :return: None
    """
    # Extract all labels as a single NumPy array
    all_labels = np.array([label for _, label in dataset])

    # Sum the occurrences of each label (each disease)
    label_count = np.sum(all_labels, axis=0)

    # # Print out the label counts
    # for i, count in enumerate(label_count):
    #     print(f"Disease {i}: {int(count)} occurrences")

    # Check if any label is non-binary (not 0 or 1)
    invalid_labels = np.where((all_labels != 0) & (all_labels != 1))
    if len(invalid_labels[0]) > 0:
        print(f"Found invalid labels at indices: {invalid_labels}")
        # Terminate the program if labels are inconsistent
        sys.exit("Error: Found non-binary labels. Terminating the program.")

def _remove_duplicate_values(dataset):
    images = [image.numpy().flatten() for image, _ in dataset]

    _, unique_indices = np.unique(images, axis=0, return_index=True)

    unique_dataset = Subset(dataset, unique_indices.tolist())

    return unique_dataset

def _noise_removal(dataset, blur_size=3):
    gaussian_blur = transforms.GaussianBlur(kernel_size=(blur_size, blur_size), sigma=(0.1, 2.0))

    # DataLoader to help batch processing
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    # Get all the images in a single batch
    images, labels = next(iter(loader))  # Extract images and labels from the dataset

    # Apply Gaussian blur to all images (this applies it to the whole tensor at once)
    blurred_images = gaussian_blur(images)
    # Create a new TensorDataset with the blurred images and the original labels
    blurred_dataset = TensorDataset(blurred_images, labels)

    return blurred_dataset

def clean_data(dataset):
    _check_labels(dataset)

    dataset = _remove_duplicate_values(dataset)
    dataset = _noise_removal(dataset)

    return dataset