# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 16:01:45 2021

@author: CEOSpace
"""
import os
import click
import json
from script import script
import numpy as np
@click.command()
@click.argument('dataset_name',default='eurosat', type=click.Choice(['mnist', 'cifar10','cifar10_out', 'eurosat', 'eurosat_out', 'eurosat_MS']))
@click.argument('net_name',default='cifar10_LeNet', type=click.Choice(['mnist_LeNet', 'cifar10_LeNet','scatter_LeNet2', 'scatter_LeNet3','scatter_LeNet4','cifar10_LeNet_MS']))
@click.argument('xp_path',default='D:/Omid/Deep-SVDD/imgs/UCM256/b1000', type=click.Path(exists=True))
@click.argument('data_path',default='D:/Omid/Deep-SVDD-PyTorch-master/data', type=click.Path(exists=True))
@click.argument('train_path',default='D:/Omid/Datasets/UCM/train.csv', type=click.Path(exists=True))  
@click.argument('test_path',default='D:/Omid/Datasets/UCM/test.csv', type=click.Path(exists=True))  
@click.option('--load_config', type=click.Path(exists=True), default=None,help='Config JSON-file path (default: None).')
@click.option('--load_model', type=click.Path(exists=True), default='D:/Omid/Deep-SVDD/imgs/UCM256/b1000/0/1/model.tar',
              help='Model file path (default: None).')
@click.option('--objective', type=click.Choice(['one-class', 'soft-boundary']), default='one-class',
              help='Specify Deep SVDD objective ("one-class" or "soft-boundary").')
@click.option('--nu', type=float, default=1024*1024, help='Deep SVDD hyperparameter nu (must be 0 < nu <= 1).')
# @click.option('--J', type=int, default=3)
@click.option('--device', type=str, default='cuda', help='Computation device to use ("cpu", "cuda", "cuda:2", etc.).')
@click.option('--seed', type=int, default=2, help='Set seed. If -1, use randomization.')
@click.option('--optimizer_name', type=click.Choice(['adam', 'amsgrad']), default='adam',
              help='Name of the optimizer to use for Deep SVDD network training.')
@click.option('--lr', type=float, default=0.0001,
              help='Initial learning rate for Deep SVDD network training. Default=0.001')
@click.option('--n_epochs', type=int, default=100, help='Number of epochs to train.')
@click.option('--lr_milestone', type=int, default=0, multiple=True,
              help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
@click.option('--batch_size', type=int, default=200, help='Batch size for mini-batch training.')
@click.option('--weight_decay', type=float, default=1e-6,
              help='Weight decay (L2 penalty) hyperparameter for Deep SVDD objective.')
@click.option('--pretrain', type=bool, default=False,
              help='Pretrain neural network parameters via autoencoder.')
@click.option('--ae_optimizer_name', type=click.Choice(['adam', 'amsgrad']), default='adam',
              help='Name of the optimizer to use for autoencoder pretraining.')
@click.option('--ae_lr', type=float, default=0.0001,
              help='Initial learning rate for autoencoder pretraining. Default=0.001')
@click.option('--ae_n_epochs', type=int, default=50, help='Number of epochs to train autoencoder.')
@click.option('--ae_lr_milestone', type=int, default=0, multiple=True,
              help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
@click.option('--ae_batch_size', type=int, default=200, help='Batch size for mini-batch autoencoder training.')
@click.option('--ae_weight_decay', type=float, default=1e-6,
              help='Weight decay (L2 penalty) hyperparameter for autoencoder objective.')
@click.option('--n_jobs_dataloader', type=int, default=0,
              help='Number of workers for data loading. 0 means that the data will be loaded in the main process.')
@click.option('--normal_class', type=int, default=8,
              help='Specify the normal class of the dataset (all other classes are considered anomalous).')

def experiment(dataset_name, net_name, xp_path, data_path, train_path, test_path, load_config, load_model, objective, nu, device, seed,
         optimizer_name, lr, n_epochs, lr_milestone, batch_size, weight_decay, pretrain, ae_optimizer_name, ae_lr,
         ae_n_epochs, ae_lr_milestone, ae_batch_size, ae_weight_decay, n_jobs_dataloader, normal_class):
    
    exps = range(0,21)
    seeds = range(1, seed)
    for n in exps:
        for seed in seeds:
            path = xp_path + '/' + str(n) + '/' + str(seed)+'/'
            load_model = xp_path + '/' + str(n) + '/' + str(seed)+'/model.tar'
            os.makedirs(path,exist_ok = True)
            script.main(dataset_name, net_name, path, data_path, train_path, test_path, load_config, load_model, objective, nu, device, seed,
                  optimizer_name, lr, n_epochs, lr_milestone, batch_size, weight_decay, pretrain, ae_optimizer_name, ae_lr,
                  ae_n_epochs, ae_lr_milestone, ae_batch_size, ae_weight_decay, n_jobs_dataloader, n)


if __name__ == '__main__':
    experiment()        

