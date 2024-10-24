# from medmnist import ChestMNIST
# import numpy as np
# import torchvision.transforms as transforms
# from torch.utils.data import Subset, DataLoader
# from data.data_preparation import clean_data
# import torch

# class DataLoaders:
#     def __init__(self):
#         # Dataset Configuration
#         self.DataClass = ChestMNIST

#         # Data Augmentation and Normalization
#         self.transform = transforms.Compose([
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomRotation(10),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[.5], std=[.5])
#         ])

#     def get_test_loader(self, testing_subset_size=100, batch_size=10):
#         test_dataset = self.DataClass(split='test', transform=self.transform, download=True)
#         test_dataset = clean_data(test_dataset)

#         test_indices = np.random.choice(np.arange(len(test_dataset)), testing_subset_size, replace=False)
#         test_subset = Subset(test_dataset, test_indices)

#         test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
#         return test_loader
    
#     def flip_label(self, label):
#         # Ensure the label is a tensor
#         if not isinstance(label, torch.Tensor):
#             label = torch.tensor(label)

#         # Flip each element in the multi-hot encoded label
#         flipped_label = torch.where(label == 0, torch.tensor(1), torch.tensor(0))
#         return flipped_label
    
#     # def get_train_loader(self, training_subset_size=100, batch_size=10, poised=False, flip_ratio=0.75):
#     #     training_dataset = self.DataClass(split='train', transform=self.transform, download=True)
#     #     training_dataset = clean_data(training_dataset)

#     #     # Subsample the training data
#     #     train_indices = np.random.choice(np.arange(len(training_dataset)), training_subset_size, replace=False)
#     #     training_subset = Subset(training_dataset, train_indices)

#     #     # If poised is True, apply label flipping
#     #     if poised:
#     #         num_samples_to_flip = int(flip_ratio * training_subset_size)  # Number of samples to flip
#     #         flip_indices = np.random.choice(np.arange(training_subset_size), num_samples_to_flip, replace=False)

#     #         # Flip the labels of the selected samples
#     #         for idx in flip_indices:
#     #             original_label = training_subset[idx][1]  # Get the original label (assuming (data, label) structure)
#     #             flipped_label = self.flip_label(original_label)  # Function to flip the label
#     #             training_subset.dataset.labels[train_indices[idx]] = flipped_label  # Apply the flipped label

#     #     # Create the data loader
#     #     train_loader = DataLoader(training_subset, batch_size=batch_size, shuffle=True)
#     #     return train_loader

#     def flip_label(self, label):
#         # Flip each element in the multi-hot encoded label
#         flipped_label = np.where(label == 0, 1, 0)
#         return flipped_label


#     def get_train_loader(self, training_subset_size=100, batch_size=10, poised=False, flip_ratio=0.75):
#         training_dataset = self.DataClass(split='train', transform=self.transform, download=True)
#         training_dataset = clean_data(training_dataset)

#         # Subsample the training data
#         train_indices = np.random.choice(np.arange(len(training_dataset)), training_subset_size, replace=False)
#         training_subset = Subset(training_dataset, train_indices)

#         # If poisoned is True, apply label flipping
#         if poised:
#             num_samples_to_flip = int(flip_ratio * training_subset_size)  # Number of samples to flip
#             flip_indices = np.random.choice(np.arange(training_subset_size), num_samples_to_flip, replace=False)

#             # Access the labels tensor
#             labels_tensor = training_subset.dataset.tensors[1]

#             # Convert labels to a mutable format (numpy array)
#             labels_array = labels_tensor.clone().detach().numpy()

#             # Flip the labels of the selected samples
#             for idx in flip_indices:
#                 label_idx = train_indices[idx]
#                 original_label = labels_array[label_idx]  # Get the original label
#                 flipped_label = self.flip_label(original_label)  # Function to flip the label
#                 labels_array[label_idx] = flipped_label  # Apply the flipped label

#             # Update the labels in the dataset
#             # Convert the modified labels back to a tensor
#             new_labels_tensor = torch.from_numpy(labels_array)
#             # Create a new TensorDataset with updated labels
#             training_subset.dataset = torch.utils.data.TensorDataset(
#                 training_subset.dataset.tensors[0], new_labels_tensor
#             )

#         # Create the data loader
#         train_loader = DataLoader(training_subset, batch_size=batch_size, shuffle=True)
#         return train_loader


from medmnist import ChestMNIST
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader, TensorDataset
from data.data_preparation import clean_data
import torch

class DataLoaders:
    def __init__(self):
        # Dataset Configuration
        self.DataClass = ChestMNIST

        # Data Augmentation and Normalization
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])

    def get_test_loader(self, testing_subset_size=1000, batch_size=10):
        test_dataset = self.DataClass(split='test', transform=self.transform, download=True)
        test_dataset = clean_data(test_dataset)

        test_indices = np.random.choice(len(test_dataset), testing_subset_size, replace=False)
        test_subset = Subset(test_dataset, test_indices)

        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
        return test_loader

    def flip_labels(self, labels, flip_ratio=0.75):
        # labels: numpy array of shape (num_samples, num_labels)
        num_samples, num_labels = labels.shape
        num_labels_to_flip = int(flip_ratio * num_labels)

        # For each sample, select indices to flip
        flip_indices = np.array([
            np.random.choice(num_labels, num_labels_to_flip, replace=False)
            for _ in range(num_samples)
        ])

        # Create a mask for the labels to flip
        flip_mask = np.zeros_like(labels, dtype=bool)
        row_indices = np.arange(num_samples)[:, np.newaxis]
        flip_mask[row_indices, flip_indices] = True

        # Flip the labels using the mask
        labels[flip_mask] = 1 - labels[flip_mask]

        return labels

    def get_train_loader(self, training_subset_size=50000, batch_size=10, poisoned=False, flip_ratio=0.75):
        training_dataset = self.DataClass(split='train', transform=self.transform, download=True)
        training_dataset = clean_data(training_dataset)

        # Access the data and labels from the TensorDataset
        data_tensor = training_dataset.tensors[0]
        labels_tensor = training_dataset.tensors[1]

        # Subsample the training data
        train_indices = np.random.choice(len(training_dataset), training_subset_size, replace=False)

        # If poisoned is True, apply label flipping
        if poisoned:
            # Extract labels for the selected indices
            labels = labels_tensor[train_indices].numpy()
            labels = labels.copy()  # Make a copy to avoid modifying the original labels

            # Flip labels efficiently
            labels = self.flip_labels(labels, flip_ratio=flip_ratio)

            # Update the labels tensor with flipped labels
            labels_tensor[train_indices] = torch.from_numpy(labels)

        # Create a new TensorDataset with updated labels
        poisoned_dataset = TensorDataset(data_tensor, labels_tensor)

        # Create the subset using the poisoned dataset
        training_subset = Subset(poisoned_dataset, train_indices)

        # Create the data loader
        train_loader = DataLoader(training_subset, batch_size=batch_size, shuffle=True)
        return train_loader
