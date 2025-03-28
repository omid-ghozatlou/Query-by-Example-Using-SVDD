import click
import torch
import logging
import random
import numpy as np
import pandas as pd
from PIL import Image
from utils.config import Config
from utils.visualization.plot_images_grid import plot_images_grid
from deepSVDD import DeepSVDD
from datasets.main import load_dataset

class script(object):
    def main(dataset_name, net_name, xp_path, data_path, train_path, test_path, load_config, load_model, objective, nu, device, seed,
             optimizer_name, lr, n_epochs, lr_milestone, batch_size, weight_decay, pretrain, ae_optimizer_name, ae_lr,
             ae_n_epochs, ae_lr_milestone, ae_batch_size, ae_weight_decay, n_jobs_dataloader, normal_class):
        """
        Deep SVDD, a fully deep method for anomaly detection.
    
        :arg DATASET_NAME: Name of the dataset to load.
        :arg NET_NAME: Name of the neural network to use.
        :arg XP_PATH: Export path for logging the experiment.
        :arg DATA_PATH: Root path of data.
        """
    
        # Get configuration
        cfg = Config(locals().copy())
    
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_file = xp_path + '/log.txt'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        # logger.info('Dataset: %s' % dataset_name)
        logger.info('Normal class: %d' % normal_class)
        logger.info('Network: %s' % net_name)
    
        # If specified, load experiment config from JSON-file
        if load_config:
            cfg.load_config(import_json=load_config)
            # logger.info('Loaded configuration from %s.' % load_config)

        # Set seed
        if cfg.settings['seed'] != -1:
            random.seed(cfg.settings['seed'])
            np.random.seed(cfg.settings['seed'])
            torch.manual_seed(cfg.settings['seed'])
            torch.cuda.manual_seed(cfg.settings['seed'])
            torch.cuda.manual_seed_all(cfg.settings['seed'])
            torch.backends.cudnn.deterministic = True 
            torch.backends.cudnn.benchmark = False
            logger.info('Set seed to %d.' % cfg.settings['seed'])
    
        # Default device to 'cpu' if cuda is not available
        if not torch.cuda.is_available():
            device = 'cpu'
    
        # Load data
        dataset = load_dataset(dataset_name, data_path,train_path, test_path, normal_class)
        # dataset = load_dataset(dataset_name, data_path, normal_class)
        # Initialize DeepSVDD model and set neural network \phi
        deep_SVDD = DeepSVDD(cfg.settings['objective'], cfg.settings['nu'])
        deep_SVDD.set_network(net_name)
        # If specified, load Deep SVDD model (radius R, center c, network weights, and possibly autoencoder weights)
        if load_model:
            deep_SVDD.load_model(model_path=load_model, load_ae=False)

        if pretrain:
            # Pretrain model on dataset (via autoencoder)
            deep_SVDD.pretrain(dataset,
                               optimizer_name=cfg.settings['ae_optimizer_name'],
                               lr=cfg.settings['ae_lr'],
                               n_epochs=cfg.settings['ae_n_epochs'],
                               lr_milestones=cfg.settings['ae_lr_milestone'],
                               batch_size=cfg.settings['ae_batch_size'],
                               weight_decay=cfg.settings['ae_weight_decay'],
                               # J=J,
                               device=device,
                               n_jobs_dataloader=n_jobs_dataloader)
            deep_SVDD.save_ae_model(export_model=xp_path + '/ae_model.tar')
    
        # Test model
        deep_SVDD.test( dataset, device=device, n_jobs_dataloader=n_jobs_dataloader,path=xp_path)
    
        # Plot most anomalous and most normal (within-class) test samples
        indices, labels, scores = zip(*deep_SVDD.results['test_scores'])
        indices, labels, scores = np.array(indices), np.array(labels), np.array(scores)
        # idx_sorted = indices[labels == 0][np.argsort(scores[labels == 0])]  # sorted from lowest to highest anomaly score
        idx_sorted = indices[np.argsort(scores)]  # sorted from lowest to highest anomaly score
        # idx_sorted = indices[np.argsort(scores)[::-1]] #highest entropy
        if dataset_name in ('mnist', 'cifar10','eurosat','eurosat_MS'):
    
            if dataset_name == 'mnist':
                X_normals = dataset.test_set.test_data[idx_sorted[:32], ...].unsqueeze(1)
                X_outliers = dataset.test_set.test_data[idx_sorted[-32:], ...].unsqueeze(1)
    
            if dataset_name == 'cifar10':
                X_normals = torch.tensor(np.transpose(dataset.test_set.data[idx_sorted[:32], ...], (0, 3, 1, 2)))
                X_outliers = torch.tensor(np.transpose(dataset.test_set.data[idx_sorted[-32:], ...], (0, 3, 1, 2)))
            if dataset_name == 'eurosat':
                path='D:/Omid/Datasets/UCM/test.csv'
                test_d = pd.read_csv(path)
                imgs=[]
                imgs2=[]
                for f in range(60):
                    img=Image.open(test_d.iloc[idx_sorted[f],0]).convert('RGB')
                    img=img.resize(256,256,3)
                    imgs.append(np.transpose(np.array(img),(2,0,1)))   
                    # imgs2.append(np.transpose(np.array(Image.open(test_d.iloc[idx_sorted[len(idx_sorted)-1-f],0]).convert('RGB')),(2,0,1)))
                    
                X_normals = torch.tensor(imgs)
                # X_outliers = torch.tensor(imgs2)
                # X_outliers = torch.tensor(np.transpose(dataset.test_set[idx_sorted[-32:], ...], ( 1, 2,0)))
            plot_images_grid(X_normals, export_img=xp_path + '/normals of %d class1' % normal_class, title='Most relevant examples of %d class' % normal_class, padding=2)
            # plot_images_grid(X_outliers, export_img=xp_path + '/outliers of %d class1' % normal_class, title='Most ambiguous examples of %d class' % normal_class, padding=2)
            out_file = open(xp_path+ '/sort.txt','w')
            for i in range(len(idx_sorted)):
                out_file.write('%d \t %s \t %d \n' %( idx_sorted[i], test_d.iloc[idx_sorted[i],0],test_d.iloc[idx_sorted[i],1]))
            out_file.close()

