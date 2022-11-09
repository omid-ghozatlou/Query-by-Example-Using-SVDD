import os
import cv2
import gc
import time
from keras.applications.imagenet_utils import preprocess_input
import numpy as np  
np.set_printoptions(threshold=np.inf)
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.keras import backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

from keras import layers, activations
from keras.models import Model
from keras.layers import Input,Conv2D, Dense,Flatten,Dropout,ZeroPadding2D,BatchNormalization,Activation,Add,Dot,AveragePooling2D,Lambda
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.convolutional import MaxPooling2D  
def read_image(image_name):
    im=cv2.imread(image_name)
    im=cv2.resize(im, (224, 224))
    return im

#数据扩充
def img_Rotation(img,angel):
    if(0 == angel):
        dst = img
    else:
        rows,cols=img.shape[:2]
        #angel度旋转
        M=cv2.getRotationMatrix2D((cols/2,rows/2),angel,1)
        dst=cv2.warpAffine(img,M,(cols,rows))
    
    return dst
dataset = 'UCM'
Extension='.tif'

# base_path='./Dataset/' + dataset
base_path= 'D:/Omid/UPB/SVM/Scattering/data/UCM'
All_Labels = os.listdir(os.path.join(base_path, 'Train'))
num_classes = len(All_Labels)
images_Train = []
labels_Train = []
images_Val = []
labels_Val = []

for i in range(0, num_classes):
    file_dir = os.path.join(base_path, 'Train', All_Labels[i])
    file_names = os.listdir(file_dir)
    for file_name in file_names:
        if(file_name.endswith(Extension)):
            finalFileName = os.path.join(file_dir, file_name)
            Label = np.linspace(0, 0, num_classes, dtype='int32')
            Label[i] = 1
            original_img = read_image(finalFileName)
            flipped_img = cv2.flip(original_img, 0)
            
            for j in [0, 45, 90, 135, 180, 225, 270, 315]: 
                images_Train.append(img_Rotation(original_img, j))
                labels_Train.append(Label)               
                images_Train.append(img_Rotation(flipped_img, j))
                labels_Train.append(Label)               
                
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "Get %d Train Images" %(len(images_Train)))
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + " np.array")
X_Train = np.array(images_Train, dtype='float32')
Y_Train = np.array(labels_Train)
del images_Train
gc.collect()

print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + " preprocess")
X_Train = preprocess_input(X_Train)

gc.collect()
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + " Done!")
ModelName = "Gabor-CA-ResNet50"
batch_size = 16
epochs = 50  
num_classes = Y_Train.shape[1]
import cv2
def Gabor(shape, dtype=None):    
    ksize = (7, 7)
    # 核尺寸
    #sigmas = [1] # [2, 4]
    # 角度
    #thetas = np.linspace(0, 2*np.pi, 8, endpoint=False) # np.linspace(0, np.pi, 4, endpoint=False)
    # 波长(间隔)
    #lambdas = np.linspace(2, 3, 6) # [8, 16, 32, 64]
    # 高度(越小，核函数图像会越高)
    #gammas = [1] # np.linspace(1, 0, 2, endpoint=False)
    # 中轴
    #psis = [0, 2*np.pi]
    
    gabors = []
    
    for i in range(0,int(64/4)):    
#     size, sigma, theta, lambda, gamma aspect ratio                 
        gf = cv2.getGaborKernel(ksize=ksize, sigma=1, theta=0, lambd=2, gamma=1, psi=0, ktype=cv2.CV_32F)
        gabors.append(gf)             
        gf = cv2.getGaborKernel(ksize=ksize, sigma=1, theta=np.pi/2, lambd=2, gamma=1, psi=0, ktype=cv2.CV_32F)
        gabors.append(gf)             
        gf = cv2.getGaborKernel(ksize=ksize, sigma=1, theta=np.pi/4, lambd=2, gamma=1, psi=0, ktype=cv2.CV_32F)
        gabors.append(gf)             
        gf = cv2.getGaborKernel(ksize=ksize, sigma=1, theta=np.pi/4*3, lambd=2, gamma=1, psi=0, ktype=cv2.CV_32F)
        gabors.append(gf)
    stacked_list = np.array([gabors])
    stacked_list = np.einsum('hijk->jkhi', stacked_list)
    
    b = K.constant(stacked_list, dtype='float32')
    F_0 = Lambda(lambda x: K.cast(x, dtype='float32'))(b)
    return F_0
bn_params = {
    'epsilon': 9.999999747378752e-06,
}

def residual_block(input_tensor, filters, reduction=16, strides=1, **kwargs):
    x = input_tensor
    residual = input_tensor

    # bottleneck
    x = Conv2D(filters // 4, (1, 1), kernel_initializer='he_uniform', strides=strides, use_bias=False)(x)
    x = BatchNormalization(**bn_params)(x)
    x = Activation('relu')(x)

    x = ZeroPadding2D(1)(x)
    x = Conv2D(filters // 4, (3, 3), kernel_initializer='he_uniform', use_bias=False)(x)
    x = BatchNormalization(**bn_params)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, (1, 1), kernel_initializer='he_uniform', use_bias=False)(x)
    x = BatchNormalization(**bn_params)(x)

    #  if number of filters or spatial dimensions changed
    #  make same manipulations with residual connection
    x_channels = K.int_shape(x)[-1]
    r_channels = K.int_shape(residual)[-1]

    if strides != 1 or x_channels != r_channels:

        residual = Conv2D(x_channels, (1, 1), strides=strides, kernel_initializer='he_uniform', use_bias=False)(residual)
        residual = BatchNormalization(**bn_params)(residual)

    # apply attention module
    x = ChannelSE(x, reduction=reduction)

    # add residual connection
    x = Add()([x, residual])

    x = Activation('relu')(x)

    return x

def ChannelSE(input_tensor, reduction=16):  
    channels = K.int_shape(input_tensor)[-1]
    x = Lambda(lambda a: K.mean(a, axis=[1, 2], keepdims=True))(input_tensor)
    x = Conv2D(channels // reduction, (1, 1), kernel_initializer='he_uniform', activation=activations.relu)(x)
    x = Conv2D(channels, (1, 1), kernel_initializer='he_uniform', activation=activations.hard_sigmoid)(x)
    return layers.Multiply()([input_tensor,x])    #给通道加权重
    
from keras.optimizers import SGD  
def GaborCAResNet(shape):
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
        
    img_input = Input(shape=shape)  
    print(img_input)
    x = ZeroPadding2D(3)(img_input)
    print(x)
    x = Conv2D(64, (7, 7), strides=2, use_bias=False, kernel_initializer='he_uniform')(x)
    print(x)
    x = BatchNormalization(**bn_params)(x)
    print(x)
    x = Activation('relu')(x)
    print(x)
    x = ZeroPadding2D(3)(x)
    print(x)
    x = Conv2D(64, (7, 7), strides=1, use_bias=False, kernel_initializer=Gabor, name='Gabor')(x)
    print(x)
    x = BatchNormalization(**bn_params, name='Gabor-Batch')(x)
    x = Activation('relu', name='Gabor-relu')(x)

    x = ZeroPadding2D(1)(x)
    x = MaxPooling2D((3, 3), strides=2)(x)
    
    # body of resnet
    filters = 128
    for i, stage in enumerate([3, 4, 6, 3]):

        # increase number of filters with each stage
        filters *= 2
        reduction = 16

        for j in range(stage):
            # decrease spatial dimensions for each stage (except first, because we have maxpool before)
            if i == 0 and j == 0:
                x = residual_block(x, filters, reduction=reduction, strides=1, is_first=True)
            elif i != 0 and j == 0:
                x = residual_block(x, filters, reduction=reduction, strides=2)
            else:
                x = residual_block(x, filters, reduction=reduction, strides=1)
    
   

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(num_classes, activation='softmax', name='output')(x)  

    model = Model(inputs=img_input,outputs=x, name=ModelName)  
    sgd = SGD(decay=0.0001,momentum=0.9)  
    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])  
    
    return model
model = GaborCAResNet(X_Train.shape[1:])
model.load_weights('./models/SeResNet50.h5', by_name=True)
model.summary()
model.layers[6].trainable = False
history = model.fit(X_Train, Y_Train, batch_size=batch_size, epochs=epochs, shuffle=True)
import time
save_folder = 'models/' + time.strftime("%Y-%m-%d", time.localtime()) + '/' + dataset
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# serialize model to JSON
#import pickle
model_json = model.to_json()
with open(os.path.join(save_folder, ModelName + ".json"), "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save(os.path.join(save_folder, ModelName + ".h5"))
print("Saved Model to disk")
import os
import time
import cv2
from keras.applications.imagenet_utils import preprocess_input

Extension='.jpg'
All_Labels = os.listdir(os.path.join(base_path, 'Test'))
images = []
labels = []
outputFileName = []

save_folder = 'FeatureMap/' + time.strftime("%Y-%m-%d", time.localtime()) + '/' + ModelName + '/' + dataset  +'/'

for i in range(0, num_classes):
    file_dir = os.path.join(base_path, 'Test', All_Labels[i])
    file_names = os.listdir(file_dir)
    for file_name in file_names:
        if(file_name.endswith(Extension)):
            finalFileName = os.path.join(file_dir, file_name)
            Label = np.linspace(0, 0, num_classes, dtype='int32')
            Label[i] = 1    
            images.append(read_image(finalFileName))
            labels.append(Label)
            outputFileName.append(os.path.join(save_folder, All_Labels[i], file_name).replace(Extension, ".txt"))

print("Get %d Test Images" %(len(images)))
output = np.array(images, dtype="float32")
output = preprocess_input(output)
import os
import time
from keras.models import Model  
OutPutLayer = Model(inputs=model.input, outputs=model.get_layer('avg_pool').output)
print("Saved FeatureMap to disk...")
OutputFeatures=[]
for i in range(0, len(output)):
    p = OutPutLayer.predict(output[i : i + 1])
    out=np.reshape(p,p.shape[1])
    OutputFeatures.append(out)
    print("\r当前输出：%d" %(i + 1), end= " ")

OutputFeatures = np.array(OutputFeatures, dtype="float") 
save_folder = 'FeatureMap/' + time.strftime("%Y-%m-%d", time.localtime()) + '/' + ModelName + '/' + dataset  +'/'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
np.savetxt(os.path.join(save_folder, dataset + '-Test-2048D.txt'), OutputFeatures)
print("\n保存完成！")
print(OutputFeatures.shape)
import os
import time
from keras.models import Model  
OutPutLayer = Model(inputs=model.input, outputs=model.get_layer('avg_pool').output)
print("Saved FeatureMap to disk...")
OutputFeatures=[]
for i in range(0, len(X_Train)):
    p = OutPutLayer.predict(X_Train[i : i + 1])
    out=np.reshape(p,p.shape[1])
    OutputFeatures.append(out)
    print("\r当前输出：%d" %(i + 1), end= " ")

OutputFeatures = np.array(OutputFeatures, dtype="float") 
save_folder = 'FeatureMap/' + time.strftime("%Y-%m-%d", time.localtime()) + '/' + ModelName + '/' + dataset  +'/'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
np.savetxt(os.path.join(save_folder, dataset + '-Train-2048D.txt'), OutputFeatures)
print("\n保存完成！")
print(OutputFeatures.shape)