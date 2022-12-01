# -*- coding: utf-8 -*-
"""
Created on Thu May 12 13:59:15 2022

@author: CEOSpaceTech
"""
F_score1 =[]
F_score2 =[]
presall =[]
P_all=[]   
for n in range(10):
    normal_class = n
    retrievd_number = 60
    test_number =[600,600,600,500,500,400,500,600,500,600]
    relevant_number = 400 #test_number[normal_class]
    lines = []

    f=open('D:/Omid/UPB/Journal/Deep-SVDD/EuroSAT/TSNE/one vs one/MS/org/'+str(normal_class)+'/1/sort.txt','r') 
    
    lines = f.readlines()
    Presicion=[]
    for k in [20,60,100,200,350]:
        count_p = 0   
        for i in range(k):
            if int(lines[i][-3])==normal_class:
                count_p +=1
        
        P = round((count_p /k),3)
        Presicion.append(P)
        pres=Presicion[0]
    P_all.append(Presicion)
    presall.append(pres)
    
    count_r = 0   
    for i in range(relevant_number):
        if int(lines[i][-3])==normal_class:
            count_r +=1
    R =round((count_r /relevant_number),3)
   
    
    # print(PR)
    F1=round((2*Presicion[0]*R)/(Presicion[0]+R+0.000000005),3)
    F2=round((5*Presicion[0]*R)/(4*Presicion[0]+R+0.000000005),3)
    F_score1.append(F1)
    F_score2.append(F2)
    print(R)
