import torch
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class Nav_Dataset(dataset.Dataset):
    def __init__(self):
        # Load data
        self.data = np.genfromtxt('training_data.csv', delimiter=',')

        # Normalize data and save scaler for inference
        self.scaler = MinMaxScaler()
        self.normalized_data = self.scaler.fit_transform(self.data)
        pickle.dump(self.scaler, open("saved/scaler.pkl", "wb"))

        # Split inputs (first 6 columns) and labels (last column)
        self.X = self.normalized_data[:, :-1].astype(np.float32)
        self.y = self.normalized_data[:, -1].astype(np.float32)

    def __len__(self):
        # Return total number of samples
        return len(self.y)

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = idx.item()

        sample = {
            'input': torch.tensor(self.X[idx], dtype=torch.float32),
            'label': torch.tensor(self.y[idx], dtype=torch.float32)
        }
        return sample


class Data_Loaders():
    def __init__(self, batch_size):
        # Load full dataset
        full_dataset = Nav_Dataset()

        # Split into train/test (80% / 20%)
        train_indices, test_indices = train_test_split(
            np.arange(len(full_dataset)),
            test_size=0.2,
            shuffle=True,
            random_state=42
        )


        train_subset = data.Subset(full_dataset, train_indices)
        test_subset = data.Subset(full_dataset, test_indices)

        self.train_loader = data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        self.test_loader = data.DataLoader(test_subset, batch_size=batch_size, shuffle=False)


def main():
    batch_size = 16
    data_loaders = Data_Loaders(batch_size)

    # Example iteration (required format)
    for idx, sample in enumerate(data_loaders.train_loader):
        _, _ = sample['input'], sample['label']
    for idx, sample in enumerate(data_loaders.test_loader):
        _, _ = sample['input'], sample['label']


if __name__ == '__main__':
    main()
