# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 11:00:32 2022

@author: CEOSpace
"""

import pandas as pd

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
############################ Center ##########
# c0=np.load('D:/Omid/Deep-SVDD/imgs/EuroSAT/MS/b100/0/c.npy')
# center_MS=np.concatenate((np.expand_dims(c0, 0),np.expand_dims(c1, 0),np.expand_dims(c2, 0),np.expand_dims(c3, 0),np.expand_dims(c4, 0),np.expand_dims(c5, 0),np.expand_dims(c6, 0),np.expand_dims(c7, 0),np.expand_dims(c8, 0),np.expand_dims(c9, 0)),axis=1)
# np.save('D:/Omid/Deep-SVDD/center_MS.npy',center_MS)
#########################################

data_info = pd.read_csv('D:/Omid/Datasets/EuroSAT_MS/test.csv')
# First column contains the image paths
image_arr = np.array(data_info.iloc[:, 0])
# Second column is the targets (labels)
target_arr = np.array(data_info.iloc[:, 1]).astype('int8')
# imgs=[]
# for f in range(len(image_arr)):
#     imgs.append(np.array(Image.open(image_arr[f]).convert('RGB')))
##########################################
# for n in range(10):
#     normal_class = n
#     path ='D:/Omid/Deep-SVDD/imgs/EuroSAT/drift/noae/b1000/'+str(normal_class)+'/1'
#     embeding = np.load(path+'/Embeding.npy')
#     Centers=np.load('D:/Omid/Deep-SVDD/center.npy')  # for RGB remove _MS
#     C = np.expand_dims(Centers[normal_class],axis=0)
    
#     em=np.reshape(embeding,(len(target_arr),128))
#     emb=np.concatenate((C,em), axis=0)
#     # Create a two dimensional t-SNE projection of the embeddings
#     tsne = TSNE(2, verbose=1, random_state=1)
#     tsne_proj = tsne.fit_transform(emb)
#     tsne_c = tsne_proj[0]
#     tsne_proj = np.delete(tsne_proj, 0, 0)
    
#     # Plot those points as a scatter plot and label them based on the pred labels
#     cmap = cm.get_cmap('tab10')
#     classes = ['AC','For','HV','High','Ind','Pas','PC','Res','Riv','Sea','Center']
#     p=0,1,2,3,4,5,6,7,8,9
#     p=np.delete(p,normal_class)
#     for j in range(9):
#         fig, ax = plt.subplots(figsize=(8,8))   
#         for lab in normal_class,p[j]:
#         # num_categories = 10
#         # for lab in range(num_categories):
#             indices = target_arr==lab
#             ax.scatter(tsne_proj[indices,0],tsne_proj[indices,1], c=np.array(cmap(lab)).reshape(1,4), label = lab ,alpha=0.5)    
        
#         ax.scatter(tsne_c[0],tsne_c[1],marker='+',c='black',alpha=0.9)
#         ax.legend([classes[normal_class],classes[p[j]],'center'],fontsize='small', markerscale=2)
#         plt.savefig(path+"/Plot_" + str(normal_class)+ ' vs '+str(p[j]) + ".png")
#         plt.show()

############################ all classes #################
path ='D:/Omid/Deep-SVDD/imgs/EuroSAT/drift/noae/b1000/0/1'
embeding = np.load(path+'/Embeding.npy')
Centers=np.load('D:/Omid/Deep-SVDD/center.npy')  # for RGB remove _MS
C = np.expand_dims(Centers[0],axis=0)

em=np.reshape(embeding,(len(target_arr),128))
emb=np.concatenate((C,em), axis=0)
# Create a two dimensional t-SNE projection of the embeddings
tsne = TSNE(2, verbose=1, random_state=1)
# tsne = TSNE(3, verbose=1, random_state=1) # 3D projection
tsne_proj = tsne.fit_transform(emb)
tsne_c = tsne_proj[0]
tsne_proj = np.delete(tsne_proj, 0, 0)
cmap = cm.get_cmap('tab10')
classes = ['A.C.','Forset','H.V.','Highway','Industrial','Pasture','P.C.','Residential','River','Sea','Center']

fig, ax = plt.subplots(figsize=(8,8))   
num_categories = 10
for lab in range(num_categories):
    indices = target_arr==lab
    ax.scatter(tsne_proj[indices,0],tsne_proj[indices,1], c=np.array(cmap(lab)).reshape(1,4), label = lab ,alpha=0.5)    

# ax.scatter(tsne_c[0],tsne_c[1],marker='+',c='black',alpha=0.9)
ax.legend(classes,fontsize='small', markerscale=2)
plt.savefig(path+"/Plotomid.png")
plt.show()
######################## #D scatter plot ################
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# num_categories = 10
# for lab in range(num_categories):
#     indices = target_arr==lab
#     p = ax.scatter3D(tsne_proj[indices,0],tsne_proj[indices,1],tsne_proj[indices,2], c=np.array(cmap(lab)).reshape(1,4),label = lab, cmap="jet")
# fig.colorbar(p, ax=ax)
