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


################################################################################
# Settings
################################################################################
# @click.command()
# @click.argument('dataset_name',default='eurosat', type=click.Choice(['mnist', 'cifar10', 'eurosat']))
# @click.argument('net_name',default='scatter_LeNet3', type=click.Choice(['mnist_LeNet', 'cifar10_LeNet','scatter_LeNet2', 'scatter_LeNet3','scatter_LeNet4','cifar10_LeNet_ELU']))
# @click.argument('xp_path',default='D:/Omid/Deep-SVDD-PyTorch-master/imgs/Euro/Scatter/J3/8', type=click.Path(exists=True))
# @click.argument('data_path',default='D:/Omid/Deep-SVDD-PyTorch-master/data', type=click.Path(exists=True))
# @click.argument('train_path',default='D:/Omid/Deep-SVDD-PyTorch-master/data/train.csv', type=click.Path(exists=True))  
# @click.argument('test_path',default='D:/Omid/Deep-SVDD-PyTorch-master/data/test.csv', type=click.Path(exists=True))  
# @click.option('--load_config', type=click.Path(exists=True), default=None,help='Config JSON-file path (default: None).')
# @click.option('--load_model', type=click.Path(exists=True), default=None,#'D:/Omid/Deep-SVDD-PyTorch-master/imgs/EuroSAT/J3/8omid/model.tar',
#               help='Model file path (default: None).')
# @click.option('--objective', type=click.Choice(['one-class', 'soft-boundary']), default='one-class',
#               help='Specify Deep SVDD objective ("one-class" or "soft-boundary").')
# @click.option('--nu', type=float, default=0.1, help='Deep SVDD hyperparameter nu (must be 0 < nu <= 1).')
# # @click.option('--J', type=int, default=3)
# @click.option('--device', type=str, default='cuda', help='Computation device to use ("cpu", "cuda", "cuda:2", etc.).')
# @click.option('--seed', type=int, default=10, help='Set seed. If -1, use randomization.')
# @click.option('--optimizer_name', type=click.Choice(['adam', 'amsgrad']), default='adam',
#               help='Name of the optimizer to use for Deep SVDD network training.')
# @click.option('--lr', type=float, default=0.0001,
#               help='Initial learning rate for Deep SVDD network training. Default=0.001')
# @click.option('--n_epochs', type=int, default=100, help='Number of epochs to train.')
# @click.option('--lr_milestone', type=int, default=0, multiple=True,
#               help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
# @click.option('--batch_size', type=int, default=200, help='Batch size for mini-batch training.')
# @click.option('--weight_decay', type=float, default=1e-6,
#               help='Weight decay (L2 penalty) hyperparameter for Deep SVDD objective.')
# @click.option('--pretrain', type=bool, default=True,
#               help='Pretrain neural network parameters via autoencoder.')
# @click.option('--ae_optimizer_name', type=click.Choice(['adam', 'amsgrad']), default='adam',
#               help='Name of the optimizer to use for autoencoder pretraining.')
# @click.option('--ae_lr', type=float, default=0.0001,
#               help='Initial learning rate for autoencoder pretraining. Default=0.001')
# @click.option('--ae_n_epochs', type=int, default=150, help='Number of epochs to train autoencoder.')
# @click.option('--ae_lr_milestone', type=int, default=0, multiple=True,
#               help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
# @click.option('--ae_batch_size', type=int, default=200, help='Batch size for mini-batch autoencoder training.')
# @click.option('--ae_weight_decay', type=float, default=1e-6,
#               help='Weight decay (L2 penalty) hyperparameter for autoencoder objective.')
# @click.option('--n_jobs_dataloader', type=int, default=0,
#               help='Number of workers for data loading. 0 means that the data will be loaded in the main process.')
# @click.option('--normal_class', type=int, default=8,
#               help='Specify the normal class of the dataset (all other classes are considered anomalous).')
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
    
        # Print arguments
        # logger.info('Log file is %s.' % log_file)
        # logger.info('Data path is %s.' % data_path)
        # logger.info('Export path is %s.' % xp_path)
    
        # logger.info('Dataset: %s' % dataset_name)
        logger.info('Normal class: %d' % normal_class)
        logger.info('Network: %s' % net_name)
    
        # If specified, load experiment config from JSON-file
        if load_config:
            cfg.load_config(import_json=load_config)
            # logger.info('Loaded configuration from %s.' % load_config)
    
        # Print configuration
        # logger.info('Deep SVDD objective: %s' % cfg.settings['objective'])
        # logger.info('nu-paramerter: %.2f' % cfg.settings['nu'])
    
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
        # logger.info('Computation device: %s' % device)
        # logger.info('Number of dataloader workers: %d' % n_jobs_dataloader)
    
        # Load data
        dataset = load_dataset(dataset_name, data_path,train_path, test_path, normal_class)
        # dataset = load_dataset(dataset_name, data_path, normal_class)
        # Initialize DeepSVDD model and set neural network \phi
        deep_SVDD = DeepSVDD(cfg.settings['objective'], cfg.settings['nu'])
        deep_SVDD.set_network(net_name)
        # If specified, load Deep SVDD model (radius R, center c, network weights, and possibly autoencoder weights)
        if load_model:
            deep_SVDD.load_model(model_path=load_model, load_ae=False)
            # deep_SVDD.load_ae_model(model_path=load_model, load_ae=False)
            # logger.info('Loading model from %s.' % load_model)
    
        # logger.info('Pretraining: %s' % pretrain)
        if pretrain:
            # Log pretraining details
            # logger.info('Pretraining optimizer: %s' % cfg.settings['ae_optimizer_name'])
            # logger.info('Pretraining learning rate: %g' % cfg.settings['ae_lr'])
            # logger.info('Pretraining epochs: %d' % cfg.settings['ae_n_epochs'])
            # logger.info('Pretraining learning rate scheduler milestones: %s' % (cfg.settings['ae_lr_milestone'],))
            # logger.info('Pretraining batch size: %d' % cfg.settings['ae_batch_size'])
            # logger.info('Pretraining weight decay: %g' % cfg.settings['ae_weight_decay'])
    
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
        # Log training details
        # logger.info('Training optimizer: %s' % cfg.settings['optimizer_name'])
        # logger.info('Training learning rate: %g' % cfg.settings['lr'])
        # logger.info('Training epochs: %d' % cfg.settings['n_epochs'])
        # logger.info('Training learning rate scheduler milestones: %s' % (cfg.settings['lr_milestone'],))
        # logger.info('Training batch size: %d' % cfg.settings['batch_size'])
        # logger.info('Training weight decay: %g' % cfg.settings['weight_decay'])
    
        # Train model on dataset
        # deep_SVDD.train(dataset,
        #                 optimizer_name=cfg.settings['optimizer_name'],
        #                 lr=cfg.settings['lr'],
        #                 n_epochs=cfg.settings['n_epochs'],
        #                 lr_milestones=cfg.settings['lr_milestone'],
        #                 batch_size=cfg.settings['batch_size'],
        #                 weight_decay=cfg.settings['weight_decay'],
        #                 # J=J,
        #                 device=device,
        #                 n_jobs_dataloader=n_jobs_dataloader)
    
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

            # for ind, scores, idx_sorted, idx_score_sorted in tuple(zip(self.trainer.ind, self.trainer.scores,self.trainer.idx_sorted, self.trainer.idx_score_sorted)):
            #     out_file.write('%d \t %.6f \n' %( idx_sorted, idx_score_sorted))
            out_file.close()
        # Save results, model, and configuration
        # deep_SVDD.save_results(export_json=xp_path + '/results.json')
        # deep_SVDD.save_model(export_model=xp_path + '/model.tar')
        # cfg.save_config(export_json=xp_path + '/config.json')
    
    
    # if __name__ == '__main__':
    #     main()
