from torch.utils.data import Dataset
import torch
import json
from collections import defaultdict
import os


class FemnistDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, train=True, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.width = 28
        self.data = torch.tensor([])
        self.targets = torch.tensor([])

        self.clients = []
        self.groups = []
        self.dict_users = {}

        # Select training set or test set
        dataset = "test"
        if train:
            dataset = "train"

        files = os.listdir(os.path.join(root_dir, dataset))
        files = [f for f in files if f.endswith('.json')]

        for f in files:

            with open(os.path.join(root_dir, dataset, f), 'r') as inf:
                cdata = json.load(inf)

            # List of clients
            self.clients.extend(cdata['users'])

            for user, data in cdata['user_data'].items():

                # Figure out the index of this data in the dataset
                start_index = len(self.data)
                end_index = start_index + len(data['x'])
                idx = list(range(start_index, end_index))

                # Extend data tensor
                self.data = torch.cat(
                    (self.data,
                     torch.reshape(torch.tensor(data['x']), (-1,  self.width,  self.width))))

                # Extend the target tensor
                self.targets = torch.cat(
                    (self.targets,
                     torch.tensor(data['y'])))

                # Check if this user already exists in the dictionary
                if user in self.dict_users:
                    self.dict_users[user].extend(idx)
                else:
                    self.dict_users[user] = idx

        self.root_dir=root_dir
        self.transform=transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx=idx.tolist()

        image=self.data[idx]
        target=self.targets[idx]

        sample=(
            torch.reshape(image, (1,  self.width,  self.width)),
            target)

        if self.transform:
            sample=self.transform(sample)

        return sample
