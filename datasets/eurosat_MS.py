import pandas as pd
from PIL import Image
import tifffile
from torch.utils.data import Dataset, Subset
from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import get_target_label_idx, global_contrast_normalization
import torch
import torchvision.transforms as transforms
import numpy as np

class EuroSAT_MS(TorchvisionDataset):

    def __init__(self, root: str,test_root: str, normal_class=5):
        super().__init__(root)

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 10))
        self.outlier_classes.remove(normal_class)

        # Pre-computed min and max values (after applying GCN) from train data per class of EuroSAT
        min_max = [ (-2.1428976, 2.55538),
                    (-2.6212, 3.0628018),
                    (-2.41149, 2.6833959),
                    (-2.2780416, 2.6324477),
                    (-1.9071723, 2.2171693),
                    (-2.2763007, 2.6110656),
                    (-2.3310583, 2.4709136),
                    (-2.0965083, 2.4113462),
                    (-2.216074, 3.0791802),
                    (-3.8719537, 7.358742)]

        # CIFAR-10 preprocessing: GCN (with L1 norm) and min-max feature scaling to [0,1]
        transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((32, 32)),
                                        transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
                                        transforms.Normalize([min_max[normal_class][0]] * 13,
                                                             [min_max[normal_class][1] - min_max[normal_class][0]] * 13)])

        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        train_set = Myeuro(root=self.root,
                              transform=transform, target_transform=target_transform)

        # train_size = int(0.8 * len(train_set))
        # test_size = len(train_set) - train_size
        # train_set, self.test_set = torch.utils.data.random_split(train_set,[train_size, test_size], generator=torch.Generator().manual_seed(1))
        
        # Subset train set to normal class
        train_idx_normal = get_target_label_idx(train_set.target_ten.clone().data.cpu().numpy(), self.normal_classes)
        train_idx_out = get_target_label_idx(train_set.target_ten.clone().data.cpu().numpy(), self.outlier_classes)
        train_out = np.random.choice(train_idx_out, int((1) * len(train_idx_normal)))
        self.train_set = Subset(train_set, train_idx_normal)
        self.val_set = Subset(train_set, train_out)
        self.test_set = Myeuro(root=test_root,
                                  transform=transform, target_transform=target_transform)


class Myeuro(Dataset):
    """Torchvision CIFAR10 class with patch of __getitem__ method to also return the index of a data sample."""

    def __init__(self, root, transform=None, target_transform=None):
        super().__init__()
        
        # Transforms
        self.to_tensor = transforms.ToTensor()
        
        # Read the csv file
        self.data_info = pd.read_csv(root)
        
        # First column contains the image paths
        self.image_arr = np.array(self.data_info.iloc[:, 0])
        
        # Second column is the targets (labels)
        self.target_arr = np.array(self.data_info.iloc[:, 1])
        self.target_ten = torch.from_numpy(self.target_arr)
        
        # Calculate len        
        self.data_len = len(self.data_info.index)
        
        #initializing transforms        
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):

        # Get image name
        image_name = self.image_arr[index]
        # Open image and convert to greyscale
        # img = Image.open(image_name).convert('RGB')
        img = tifffile.imread(image_name).astype(np.uint8)
        # Get target (label) of the image
        target = torch.from_numpy(np.array(self.target_arr[index]))

        # Transform image
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)        

        return img, target, index
    
    def __len__(self):
        return self.data_len
    