#!/usr/bin/env python 
import pandas as pd
import cv2
import math
import numpy as np
import random
import math
import tensorflow as tf
from scipy.stats import norm


import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,Lambda
from keras.layers import Conv2D, MaxPooling2D,Cropping2D
from keras.optimizers import SGD
from keras import regularizers
from keras import backend as K

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import sklearn

#get cvs data
data_path = './data/'
df = pd.read_csv('./data/driving_log.csv')

# sample data
sample_data = df[['center','left','right','steering']]
n_train = len(sample_data)

# camreas and angle offset
camera_pos = ['left','center','right']
angle_offset = {'center':0,'left':0.2,'right':0.2}


# image shape
x_image = (160,320,3)

# training data
X_train_data = []
y_train_data = []



def EqualizeHist_brightness(image):
    
    #applies histogram equilization on V channel of HSV 
    
    image_HSV = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image_HSV[:,:,2] = cv2.equalizeHist(image_HSV[:,:,2])
    image = cv2.cvtColor(image_HSV,cv2.COLOR_HSV2RGB)
    return image



# get the angle of camera images
def get_angle(camera_position,center_angle,image_index):
    
    if camera_position == 'left':
        #left camera 
        left_angle_tan = angle_offset['left'] + math.tan(center_angle)
        angle = math.atan(left_angle_tan)
    elif camera_position == 'right':
        #right camera
        right_angle_tan = math.tan(center_angle) - angle_offset['right'] 
        angle = math.atan(right_angle_tan)
    else:
        angle = center_angle
    
    if angle >= 1.0 :
        angle = 1.0
    elif angle <= -1.0:
        angle = -1.0
    
    return angle

# get traning data
def get_data():

    for i in range(n_train):
        
        for j in range(len(camera_pos)):
            image_name = data_path + sample_data[camera_pos[j]][i].strip()
            center_angle = sample_data['steering'][i]
            angel = get_angle(camera_pos[j],center_angle,i)
            X_train_data.append(image_name)
            y_train_data.append(angel) 

    return  len(y_train_data)


# get sample weights
def get_weight(y_train,num_bins=10):
    
    weights_bin = np.zeros(num_bins)
    weights = np.zeros(len(y_train))
    
    # Historgram and Gaussian distribution
    nums,bins = np.histogram(y_train,num_bins)    
    prob = norm.pdf(bins,0,0.8)

    # weight of each bin
    for i in range(num_bins):
        if nums[i]:
            weights_bin[i] = prob[i+1]
        else :
            weights_bin[i] = 0
            nums[i] = 1
    #weight of each training data
    weights_bin = weights_bin / np.sum(weights_bin)
    weights_bin = weights_bin / nums
    
    bin_index = np.digitize(y_train,bins)
       
    for i in range(len(y_train)):
        if bin_index[i] > num_bins :
            bin_index[i] -= 1
        weights[i] = weights_bin[bin_index[i]-1] 
       
    return weights,prob




# image generator 
def generator(X_train,y_train,batch_size = 32,augment = False):
    
    num_sample = len (y_train)
    X_train_index = range(num_sample)
    #get the weight of each sample
    weights = np.zeros(len(y_train),dtype = np.float32)
    weights,_ = get_weight(y_train,num_bins=50)
    
    while True:
        X_train_index,X_weights = sklearn.utils.shuffle(X_train_index,weights)
        
        #generate data for each batch 
        for offset in range(0,num_sample,batch_size):
            # select a batch samples base on the weight of each sample
            X_batch_index = np.random.choice(X_train_index,batch_size,replace=True,p=X_weights)
            
            images = np.zeros((len(X_batch_index),160,320,3),dtype = np.float32)
            angles = np.zeros((len(X_batch_index),),dtype = np.float32)

            for i in range(len(X_batch_index)):
                
                image_index = X_batch_index[i]
                
                # original data
                image_name = X_train[image_index]
                image = cv2.imread(image_name)
                image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                angle = y_train[image_index]
                
                # augment data
                if augment :
                    
                    image_name = X_train[image_index]
                    image = cv2.imread(image_name)
                    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                    angle = y_train[image_index]
                    #Applies histogram equilization on V channel of HSV  
                    if (np.random.choice([0,1]) == 0):
                        image = EqualizeHist_brightness(image)
                    
                    # randomly flip
                    if (np.random.choice([0,1]) == 0):
                        image = cv2.flip(image,1)
                        angle = - angle 
                    
                images[i] = image
                angles[i] = angle
                           
            yield (images,angles)



# build model,Nvidia model
def build_model():
    

    model = Sequential()

    #Cropping
    model.add(Cropping2D(cropping = ((55,25),(0,0)),input_shape = x_image))
    #resize images
    model.add(Lambda(lambda x: tf.image.resize_images(x,(66,200),0)))

    #normalize the image data
    model.add(Lambda(lambda x: x/255.0 - 0.5))


    model.add(Conv2D(24, (5, 5),strides=(2, 2), kernel_initializer='TruncatedNormal',activation='relu',kernel_regularizer=regularizers.l2(0.01)))
    model.add(Conv2D(36, (5, 5),strides=(2, 2), kernel_initializer='TruncatedNormal',activation='relu',kernel_regularizer=regularizers.l2(0.01)))
    model.add(Conv2D(48, (5, 5),strides=(2, 2), kernel_initializer='TruncatedNormal',activation='relu',kernel_regularizer=regularizers.l2(0.01)))

    model.add(Conv2D(64, (3, 3), kernel_initializer='TruncatedNormal',activation='relu',kernel_regularizer=regularizers.l2(0.01)))
    model.add(Conv2D(64, (3, 3), kernel_initializer='TruncatedNormal',activation='relu',kernel_regularizer=regularizers.l2(0.01)))


    model.add(Flatten())
    model.add(Dense(100,activation='relu'))
    model.add(Dense(50,activation='relu'))
    model.add(Dense(10,activation='relu'))
    model.add(Dense(1,activation = 'tanh'))

    model.compile(loss = 'mean_squared_error',optimizer = 'adam')

    return model




if __name__=="__main__":

    num_data = get_data()

    print ("Num of data : ",num_data)

    sample_data = np.stack((X_train_data,y_train_data),axis = 1)
    #split sample data to training data and validation data
    sample_data = shuffle(sample_data,random_state = 8)
    train_data,validation_data = train_test_split(sample_data,test_size = 0.2)
    #training data
    X_train = train_data[:,0]
    y_train = np.float64(train_data[:,1])
    #validation data
    X_valid = validation_data[:,0]
    y_valid = np.float64(validation_data[:,1])

    train_gen = generator(X_train,y_train,batch_size = 64,augment = True)
    valid_gen = generator(X_valid,y_valid,batch_size = 64,augment = False)

    print ("Number of training :",len(X_train))
    print ("Number of validation :",len(X_valid))

    model = build_model()

    history_object = model.fit_generator(generator=train_gen,steps_per_epoch = 2000,epochs = 10,validation_data = valid_gen,validation_steps = 400,verbose=1)

    model.save('model.h5')
    print ("Saved model ！！")











