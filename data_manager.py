import torch
from torch.utils.data import Dataset
from skimage import io
import os

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images_labels = self.path_to_dataset(root_dir)

    def __len__(self):
        return len(self.images_labels)
    
    def path_to_dataset(self, root_dir):
        dirs = os.listdir(root_dir)
        images_labels = []
        for idx, dir in enumerate(dirs):
            files = os.listdir(os.path.join(root_dir, dir))
            for file in files:
                file_path = os.path.join(root_dir, dir, file)
                images_labels.append((file_path, idx))
        return images_labels

    def __getitem__(self, idx):
        img_path = self.images_labels[idx][0]
        img_label = self.images_labels[idx][1]
        img_label = torch.tensor(img_label)
        image = io.imread(self.images_labels[idx][0])
        counter = 1
        while(image.shape[2] == 4):
            image = io.imread(self.images_labels[(idx + counter) % self.__len__()][0])
            counter += 1
        #print(image.shape)
        if self.transform:
            image = self.transform(image.copy())

        return (image, img_label, img_path)