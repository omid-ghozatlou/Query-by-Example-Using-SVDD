import os
import time
import numpy as np
np.set_printoptions(threshold=np.inf)
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.keras import backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

from keras.models import Sequential, Model
from keras.layers import Input,Dense,Dropout,Lambda,Concatenate
dataset = 'AID'
base_path='./FeatureMap/2020-11-28/Gabor-CA-ResNet50/' + dataset
ModelName = "Gabor-CA-ResNet50DNN"
batch_size = 16
epochs = 50
DNNlayers = [ 64, 8 ]

X_Train = np.loadtxt(base_path + "/" + dataset + "-Train-2048D.txt");
Y_Train = np.load('./Dataset/' + dataset + "/Y_Train.npy");

print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + " Get %d Train Images" %(len(X_Train)))  
num_classes = Y_Train.shape[1]
steps = int(X_Train.shape[1] / 8)
from keras.optimizers import SGD  
def DNN(shape):
    img_input = Input(shape=shape)  
    x = img_input
    x1 = Lambda(lambda x: x[:,0:steps])(x)
    x2 = Lambda(lambda x: x[:,steps:steps*2])(x)
    x3 = Lambda(lambda x: x[:,steps*2:steps*3])(x)
    x4 = Lambda(lambda x: x[:,steps*3:steps*4])(x)
    x5 = Lambda(lambda x: x[:,steps*4:steps*5])(x)
    x6 = Lambda(lambda x: x[:,steps*5:steps*6])(x)
    x7 = Lambda(lambda x: x[:,steps*6:steps*7])(x)
    x8 = Lambda(lambda x: x[:,steps*7:])(x)
    for n in DNNlayers:
        x1 = Dense(n, activation='relu', name='fc%d-1' %n)(x1)  
        x1 = Dropout(0.5)(x1)
    for n in DNNlayers:
        x2 = Dense(n, activation='relu', name='fc%d-2' %n)(x2)  
        x2 = Dropout(0.5)(x2)
    for n in DNNlayers:
        x3 = Dense(n, activation='relu', name='fc%d-3' %n)(x3)  
        x3 = Dropout(0.5)(x3)
    for n in DNNlayers:
        x4 = Dense(n, activation='relu', name='fc%d-4' %n)(x4)  
        x4 = Dropout(0.5)(x4)
    for n in DNNlayers:
        x5 = Dense(n, activation='relu', name='fc%d-5' %n)(x5)  
        x5 = Dropout(0.5)(x5)
    for n in DNNlayers:
        x6 = Dense(n, activation='relu', name='fc%d-6' %n)(x6)  
        x6 = Dropout(0.5)(x6)
    for n in DNNlayers:
        x7 = Dense(n, activation='relu', name='fc%d-7' %n)(x7)  
        x7 = Dropout(0.5)(x7)
    for n in DNNlayers:
        x8 = Dense(n, activation='relu', name='fc%d-8' %n)(x8)  
        x8 = Dropout(0.5)(x8)
    x = Concatenate(axis=1, name='outputFeature')([x1, x2, x3, x4, x5, x6, x7, x8])
    x = Dense(num_classes, activation='softmax', name='output')(x)  
    
    model = Model(inputs=img_input,outputs=x, name=ModelName)  
    sgd = SGD(decay=0.0001,momentum=0.9)  
    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])  
    
    return model
model = DNN(X_Train.shape[1:])
model.summary()
def SaveModel(n):
    save_folder = 'models/' + time.strftime("%Y-%m-%d", time.localtime()) + '/' + dataset + '/' + str(n)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # serialize model to JSON
    #import pickle
    model_json = model.to_json()
    with open(os.path.join(save_folder, ModelName + ".json"), "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save(os.path.join(save_folder, ModelName + ".h5"))
    #pickle.dump(history.history, open('history/UCMerced_LandUse/AlexNet.p','wb'))
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + " Saved Model to disk")
X_Test = np.loadtxt(base_path + "/" + dataset + "-Test-2048D.txt")
def SaveFeature():
    save_folder = 'FeatureMap/' + time.strftime("%Y-%m-%d", time.localtime()) + '/' + ModelName + '/' + dataset + '/'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    from keras.models import Model  
    for n in DNNlayers:
        OutPutLayer = Model(inputs=model.input, outputs=model.get_layer('fc%d' %n).output)
        print("Saved %d FeatureMap to disk..." %n)
        OutputFeatures=[]
        for i in range(0, len(X_Test)):
            p = OutPutLayer.predict(X_Test[i : i + 1])
            out=np.reshape(p,p.shape[1])
            OutputFeatures.append(out)
            print("\r当前输出：%d" %(i + 1), end= " ")

        OutputFeatures = np.array(OutputFeatures, dtype="float") 
        np.savetxt(os.path.join(save_folder, dataset + '-Test-%dD.txt' %n), OutputFeatures)
        print("\n保存完成！")
        print(OutputFeatures.shape)
        
def SaveOutputFeature():
    save_folder = 'FeatureMap/' + time.strftime("%Y-%m-%d", time.localtime()) + '/' + ModelName + '/' + dataset + '/'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    from keras.models import Model  
    OutPutLayer = Model(inputs=model.input, outputs=model.get_layer('outputFeature').output)
    print("Saved FeatureMap to disk...")
    OutputFeatures=[]
    for i in range(0, len(X_Test)):
        p = OutPutLayer.predict(X_Test[i : i + 1])
        out=np.reshape(p,p.shape[1])
        OutputFeatures.append(out)
        print("\r当前输出：%d" %(i + 1), end= " ")

    OutputFeatures = np.array(OutputFeatures, dtype="float") 
    np.savetxt(os.path.join(save_folder, dataset + '-Test-%dD.txt' %OutputFeatures.shape[-1]), OutputFeatures)
    print("\n保存完成！")
    print(OutputFeatures.shape)
history = model.fit(X_Train, Y_Train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=0)
SaveModel()
SaveOutputFeature()