from .mnist import MNIST_Dataset
from .cifar10 import CIFAR10_Dataset 
from .cifar10_out import CIFAR10_out 
from .eurosat import EuroSAT_Dataset
from .eurosat_MS import EuroSAT_MS
from .eurosat_out import EuroSAT_out

def load_dataset(dataset_name, data_path, train_path, test_path, normal_class):# for EuroSAT add test_path
    """Loads the dataset."""

    implemented_datasets = ('mnist', 'cifar10','cifar10_out','eurosat', 'eurosat_out', 'eurosat_MS')
    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name == 'mnist':
        dataset = MNIST_Dataset(root=data_path, normal_class=normal_class)

    if dataset_name == 'cifar10':
        dataset = CIFAR10_Dataset(root=data_path, normal_class=normal_class)
    
    if dataset_name == 'cifar10_out':
        dataset = CIFAR10_out(root=data_path, normal_class=normal_class)    
    if dataset_name == 'eurosat':
        dataset = EuroSAT_Dataset(root=train_path, test_root=test_path, normal_class=normal_class)
        
    if dataset_name == 'eurosat_out':
        dataset = EuroSAT_out(root=train_path, test_root=test_path, normal_class=normal_class)    
    
    if dataset_name == 'eurosat_MS':
        dataset = EuroSAT_MS(root=train_path, test_root=test_path, normal_class=normal_class)  
    
    return dataset
