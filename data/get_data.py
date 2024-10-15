from medmnist import ChestMNIST
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader
from data.data_preparation import clean_data


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

    def get_test_loader(self, testing_subset_size=100, batch_size=10):
        test_dataset = self.DataClass(split='test', transform=self.transform, download=True)
        test_dataset = clean_data(test_dataset)

        test_indices = np.random.choice(np.arange(len(test_dataset)), testing_subset_size, replace=False)
        test_subset = Subset(test_dataset, test_indices)

        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
        return test_loader


    def get_train_loader(self, training_subset_size=100, batch_size=10):
        training_dataset = self.DataClass(split='train', transform=self.transform, download=True)
        training_dataset = clean_data(training_dataset)

        train_indices = np.random.choice(np.arange(len(training_dataset)), training_subset_size, replace=False)
        training_subset = Subset(training_dataset, train_indices)

        train_loader = DataLoader(training_subset, batch_size=batch_size, shuffle=True)
        return train_loader