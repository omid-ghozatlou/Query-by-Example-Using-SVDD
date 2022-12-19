from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import roc_auc_score
from torchsummary import summary
import logging
import time
import torch
import torch.optim as optim
import numpy as np
from kymatio import Scattering2D
import matplotlib
import matplotlib.pyplot as plt
import skimage.measure
from scipy.stats import entropy
from math import log, e
import pandas as pd


# from sklearn.model_selection import KFold
import torch.nn.functional as F
class DeepSVDDTrainer(BaseTrainer):

    def __init__(self, objective, R, c, nu: float, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150,
                 lr_milestones: tuple = (), batch_size: int = 24, weight_decay: float = 1e-6,J: int =2, device: str = 'cuda',
                 n_jobs_dataloader: int = 0):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

        assert objective in ('one-class', 'soft-boundary'), "Objective must be either 'one-class' or 'soft-boundary'."
        self.objective = objective

        # Deep SVDD parameters
        self.R = torch.tensor(R, device=self.device)  # radius R initialized with 0 by default.
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.nu = nu

        # Optimization parameters
        self.warm_up_n_epochs = 10  # number of training epochs for soft-boundary Deep SVDD before radius R gets updated
        scattering = Scattering2D(J=2, shape=(32, 32), L=6, max_order=2)
        self.scattering = scattering.to(self.device)
        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None
        self.sum = 0.0
    def train(self, dataset: BaseADDataset, net: BaseNet):
        logger = logging.getLogger()
        # k=5
        # kfold = KFold(n_splits=k, shuffle=True)
        # for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset.train_set)):
        # Get fold train data loader for cross validation
        # train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        # val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
        # train_loader = torch.utils.data.DataLoader(dataset.train_set, batch_size=self.batch_size, sampler=train_subsampler)    
        pdist = torch.nn.PairwiseDistance(p=0.5)
        # Set device for network
        net = net.to(self.device)
        # summary(net,(3*81,8,8))
        # Get train data loader
        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                               amsgrad=self.optimizer_name == 'amsgrad')

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            # logger.info('Initializing center c...')
            self.c = self.init_center_c(train_loader, net)
            
            # logger.info('Center c %s initialized.' %self.c[0])

        # Training
        # self.reset_weights(net) # K-fold cross validation
        # logger.info('Starting training...')
        # start_time = time.time()
        loss_values = list()
        net.train()
        for epoch in range(self.n_epochs):

            
            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:
                inputs, _, _ = data
                inputs = inputs.to(self.device)
                # inputs= torch.reshape(self.scattering(inputs), (inputs.shape[0],9*3,16,16)) #J=1
                # inputs= torch.reshape(self.scattering(inputs), (inputs.shape[0],49*3,8,8)) #J=2             
                # inputs= torch.reshape(self.scattering(inputs), (inputs.shape[0],217*3,4,4)) 
                #inputs= torch.reshape(self.scattering(inputs), (inputs.shape[0],417*3,2,2)) #J=4
                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = net(inputs)
                
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                # outputs=outputs.to(self.device)
                # arr=self.cov(outputs)
                # var_dist = (torch.sum(arr**2) - torch.sum((torch.diagonal(arr))**2))/(outputs.shape[1])
                # print('variance',var_dist)
                # dist = torch.sum(abs(outputs - self.c), dim=1) #L1
                # dist = torch.sum((abs(outputs - self.c))**(0.9), dim=1) #L0.5
                # dist = pdist(outputs , self.c) #L0.5
                # logger.info(outputs)
                if self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                    loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
                else:
                    loss = torch.mean(dist) 
                    # loss = torch.mean(dist) + self.nu * var_dist
                loss.backward()
                optimizer.step()

                # Update hypersphere radius R on mini-batch distances
                if (self.objective == 'soft-boundary') and (epoch >= self.warm_up_n_epochs):
                    self.R.data = torch.tensor(get_radius(dist, self.nu), device=self.device)

                loss_epoch += loss.item()
                n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
                        .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches))
        
        #     loss_values.append(loss_epoch / n_batches)
        # plt.plot(loss_values,label='Training Loss')
        # plt.savefig('D:/Omid/Deep-SVDD-PyTorch-master/imgs/train_loss_MS_32.png')
        # plt.close()
        scheduler.step()
        # self.AUC = self.val(dataset, net,val_subsampler, fold)
        # self.sum += self.AUC
        # self.train_time = time.time() - start_time
        # logger.info('Training time: %.3f' % self.train_time)

        # logger.info('Finished training.')
        # logger.info(  'Val average AUC: {:.2f}%'.format(100. * self.sum/k))
        return net
    
    def val(self, dataset: BaseADDataset, net: BaseNet, val_subsampler, fold):
        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        # Get test data loader
        val_loader = torch.utils.data.DataLoader(dataset.train_set, batch_size=self.batch_size, sampler=val_subsampler)
        val_loader2 = torch.utils.data.DataLoader(dataset.val_set, batch_size=self.batch_size)

        # Testing
        logger.info('Starting Validation fold %d:' %fold)
        # start_time = time.time()
        idx_label_score = []
        net.eval()
        with torch.no_grad():
            for data in val_loader:
                inputs, labels, idx = data
                inputs = inputs.to(self.device)
                outputs = net(inputs)
                # outputs = net(torch.reshape(self.scattering(inputs), (8,243,8, 8)))
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                if self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                else:
                    scores = dist
                # Save triples of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))
            for data in val_loader2:
                inputs, labels, idx = data
                inputs = inputs.to(self.device)
                outputs = net(inputs)
                # outputs = net(torch.reshape(self.scattering(inputs), (8,243,8, 8)))
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                if self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                else:
                    scores = dist
                # Save triples of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

        # self.test_time = time.time() - start_time
        # logger.info('Validation time for fold %d: %.3f ' %(fold, self.test_time))

        self.test_scores = idx_label_score

        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)

        self.test_auc = roc_auc_score(labels, scores)
        
        logger.info('Validation set AUC: {:.2f}%'.format(100. * self.test_auc))

        # logger.info('Finished Validation fold %d:' %fold)
        return self.test_auc
    
    def test(self, dataset: BaseADDataset, net: BaseNet):
        logger = logging.getLogger()
        pdist = torch.nn.PairwiseDistance(p=0.5)
        # Set device for network
        net = net.to(self.device)

        # Get test data loader
        _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Testing
        # logger.info('Starting testing...')
        start_time = time.time()
        idx_label_score = []
        out=[]
        net.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, idx = data
                
                inputs = inputs.to(self.device)
                # inputs= torch.reshape(self.scattering(inputs), (inputs.shape[0],9*3,16,16)) #J=1
                # inputs= torch.reshape(self.scattering(inputs), (inputs.shape[0],49*3,8,8))#J=2
                # inputs= torch.reshape(self.scattering(inputs), (inputs.shape[0],217*3,4,4))
                #inputs= torch.reshape(self.scattering(inputs), (inputs.shape[0],417*3,2,2)) #J=4
                outputs = net(inputs)
                out.append(np.array(outputs.cpu()))
                # arr=np.cov(np.transpose((outputs.cpu().detach().numpy())))
                # var_dist = (np.sum(arr**2) - sum((np.diagonal(arr))**2))/(outputs.shape[1])
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                # dist = 1- F.cosine_similarity(outputs,torch.unsqueeze(self.c,0))             
                # dist = torch.sum(abs(outputs - self.c), dim=1) #L1
                # dist = torch.sum((abs(outputs - self.c))**(0.9), dim=1) #L0.5
                # dist = pdist(outputs , self.c)
                if self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                else:
                    scores = dist
                    # print(np.array(scores).shape)
                    # scores = np.array(dist) # entropy
                    # scores = dist + self.nu * var_dist

                # Save triples of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores))
                                            # scores.cpu().data.numpy().tolist()))
        self.test_time = time.time() - start_time
        # logger.info('Testing time: %.3f' % self.test_time)

        self.test_scores = idx_label_score

        # Compute AUC
        ind, labels, scores = zip(*idx_label_score)
        self.labels = np.array(labels)
        self.scores = np.array(scores)
        # ind, scores = zip(*idx_score)
        self.ind = np.array(ind)
        # self.scores = np.array(scores)
        self.idx_sorted = self.ind[np.argsort(self.scores)]
        self.idx_score_sorted = np.take_along_axis(self.scores,self.idx_sorted,axis=0)
        self.test_auc = roc_auc_score(self.labels, self.scores)
        logger.info('Test set AUC: {:.2f}%'.format(100. * self.test_auc))
        return self.idx_sorted, out
        # logger.info('Finished testing.')

    def init_center_c(self, train_loader: DataLoader, net: BaseNet, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros(net.rep_dim, device=self.device)

        net.eval()
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                inputs, _, _ = data
                inputs = inputs.to(self.device)
                # inputs= torch.reshape(self.scattering(inputs), (inputs.shape[0],9*3,16,16)) #J=1
                # inputs= torch.reshape(self.scattering(inputs), (inputs.shape[0],49*3,8,8)) #J=2
                # inputs= torch.reshape(self.scattering(inputs), (inputs.shape[0],217*3,4,4))
                #inputs= torch.reshape(self.scattering(inputs), (inputs.shape[0],417*3,2,2)) #J=4
                outputs = net(inputs)
                              
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        c /= n_samples
        # print('input min & max:%s & %s' %(np.min((inputs).tolist()) ,np.max((inputs).tolist()))) 
        # print(((inputs).shape)) 
        # print('Scattered input max & min:%s & %s' %(np.max(self.scattering(inputs).tolist()) ,np.min(self.scattering(inputs).tolist()))) 
        # print('\n output min & max:%s & %s' %(np.min((outputs).tolist()) ,np.max((outputs).tolist())))
        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c
    def cov(self,m, rowvar=False):
        '''Estimate a covariance matrix given data.
    
        Covariance indicates the level to which two variables vary together.
        If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
        then the covariance matrix element `C_{ij}` is the covariance of
        `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.
    
        Args:
            m: A 1-D or 2-D array containing multiple variables and observations.
                Each row of `m` represents a variable, and each column a single
                observation of all those variables.
            rowvar: If `rowvar` is True, then each row represents a
                variable, with observations in the columns. Otherwise, the
                relationship is transposed: each column represents a variable,
                while the rows contain observations.
    
        Returns:
            The covariance matrix of the variables.
        '''
        if m.dim() > 2:
            raise ValueError('m has more than 2 dimensions')
        if m.dim() < 2:
            m = m.view(1, -1)
        if not rowvar and m.size(0) != 1:
            m = m.t()
        # m = m.type(torch.double)  # uncomment this line if desired
        fact = 1.0 / (m.size(1) - 1)
        m -= torch.mean(m, dim=1, keepdim=True)
        mt = m.t()  # if complex: mt = m.t().conj()
        return fact * m.matmul(mt).squeeze()
    
    def entropy2(self,labels, base=None):
          """ Computes entropy of label distribution. """
        
          n_labels = len(labels)
        
          if n_labels <= 1:
            return 0
        
          value,counts = np.unique(labels, return_counts=True)
          probs = counts / n_labels
          n_classes = np.count_nonzero(probs)
        
          if n_classes <= 1:
            return 0
        
          ent = 0.
        
          # Compute entropy
          base = e if base is None else base
          for i in probs:
            ent -= i * log(i, base)
        
          return ent
      
    def entropy3(self,labels, base=None):
          vc = pd.Series(labels).value_counts(normalize=True, sort=False)
          base = e if base is None else base
          return -(vc * np.log(vc)/np.log(base)).sum()
    
    def reset_weights(self,m):
      '''
        Try resetting model weights to avoid
        weight leakage.
      '''
      for layer in m.children():
       if hasattr(layer, 'reset_parameters'):
        # print(f'Reset trainable parameters of layer = {layer}')
        layer.reset_parameters()
def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)
