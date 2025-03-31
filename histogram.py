# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 18:58:10 2022

@author: CEOSpace
"""
import numpy as np
import matplotlib.pyplot as plt
file='D:/Omid/Deep-SVDD/imgs/EuroSAT/drift/0.001first/4/1/sorted.txt'
with open(file) as f:
    lines = f.readlines()    
scores=[]    
for i in range(5400):
    scores.append(float(lines[i][-11:])) 
    
b = plt.hist(scores, bins=160,range=(0,40))
# b = plt.hist(scores, bins='auto')
