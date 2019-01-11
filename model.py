# Load dependent libraries
import os
import csv
import cv2
import json
import errno
import pickle
import numpy as np
from scipy import misc
from sklearn.utils import shuffle
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten, Lambda, Activation, MaxPooling2D
print('Libraries loaded.')

# Declare globals
X_train= []          # features
y_train= []          # labels
correction= 0.2      # offset value to apply to left/right images
dataDir= 'data/'     # the data directory

# Function to delete an existing file
def deleteFile(file):
    try:
        os.remove(file)
    except OSError as error:
        if error.errno != errno.ENOENT:
            raise

# The following images were removed from the driving log index.
# The model is not trained on these three images.
# This function loads and preprocesses the images for a post-training test.
def loadQuicktest():
    global X_train, y_train
    fileList= ['IMGU/center_2016_12_01_13_34_05_942.jpg', 'IMGU/center_2016_12_01_13_33_56_606.jpg', 'IMGU/center_2016_12_01_13_40_49_444.jpg']
    y_train= [0.3583844, -0.401554, 0] # 8.96 deg right, 10.04 deg left, 0 straight
    X_train.append(cropResize(misc.imread(dataDir+fileList[0])))
    X_train.append(cropResize(misc.imread(dataDir+fileList[1])))
    X_train.append(cropResize(misc.imread(dataDir+fileList[2])))
    X_train= np.array(X_train)
    y_train= np.array(y_train)

# Function to crop, blur, convert color space, and resize an image
# Input array must be 160x320 (rows by columns) BGR formatted image
# Output array is 18x80
def cropResize(image):
    imgNew= image[62:134,:]
    imgNew= cv2.bilateralFilter(imgNew,7,0,75)
    imagePlex= (cv2.cvtColor(imgNew, cv2.COLOR_BGR2HSV))[:,:,1]
    return misc.imresize(imagePlex, (18,80,1))

# Function to read the driving log data, preprocess, and save to a pickle file.
# Data is loaded to the global variables
def loadData(baseName):
    global X_train, y_train
    cnt= 0
    try:
        # if a pickle file exists, fetch the dataset.
        with open(dataDir+baseName+'.pickle', mode='rb') as f:
            print(dataDir+baseName+'.pickle')
            train= pickle.load(f)
        X_train.extend(train['train_dataset'])
        y_train.extend(train['train_labels'])
        print('Data loaded from: ', baseName+'.pickle')
    except Exception as e:
        print(e)
        fileName= dataDir+'driving_log'+baseName+'.csv'
        print('Loading data from: ',fileName)
        print('[',end='',flush=True)
        with open(fileName, 'r') as csvfile:
            csvReader= csv.reader(csvfile)
            for row in csvReader:
                cnt += 1
                # crop and resize...
                img= cropResize(misc.imread(dataDir+row[0]))
                angle= float(row[3])
                # skip half of the "straight" images...
                if angle==0.0 and cnt%2==0:
                    continue
                
                if cnt%2==0:
                    # append directly the even frames...
                    X_train.append(img)
                    y_train.append(angle)
                else:
                    # flip the odd frames left-right for symmetry
                    X_train.append(np.fliplr(img))
                    y_train.append(-1.0 *angle)
                
                # conditionally load left and right images:
                if abs(angle)>0.15:
                    img= cropResize(misc.imread(dataDir+row[1].strip()))
                    # handle images like above, and apply "correction" to angles
                    if cnt%2==0:
                        X_train.append(img)
                        y_train.append(angle+correction)
                    else:
                        X_train.append(np.fliplr(img))
                        y_train.append(-1.0 * (angle+correction))
                    
                    img= cropResize(misc.imread(dataDir+row[2].strip()))
                    if cnt%2==0:
                        X_train.append(img)
                        y_train.append(angle-correction)
                    else:
                        X_train.append(np.fliplr(img))
                        y_train.append(-1.0 * (angle-correction))
                
                if cnt%100==0: print('=',end='',flush=True)
        print('] Rows:', cnt)
        pickle_file= dataDir+baseName+'.pickle'
        try:
          # save the processed data to a pickle file
          f= open(pickle_file, 'wb')
          save= {
            'train_dataset': X_train,
            'train_labels': y_train
            }
          pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
          f.close()
          print('Dataset saved to pickle file:', pickle_file)
        except Exception as e:
          print('Unable to save data to', pickle_file, ':', e)
          raise

# Create the network model
# Returns a sequential Keras model
def createModel():
    model= Sequential([Lambda(lambda x: x/127.5 - 1.0, input_shape=(18,80,1), name='Normalization')])
    model.add(Convolution2D(20, 3,12, border_mode='same', activation='relu', subsample=(2, 3),name='Convolution'))
    model.add(MaxPooling2D((2,6),(2,4),'same',name='MaxPool')) #poolsize,stride
    model.add(Dropout(0.22,name='Dropout')) #drop 22%
    model.add(Flatten(name='Flatten'))
    model.add(Dense(1, name='Output'))
    model.summary()
    model.compile(optimizer='adam', loss='mse', metrics=['mean_squared_error'])
    #plot(model, to_file='model.png', show_shapes=True)
    print('Model compiled.')
    return model

# Define a generator to return a batch of training data 
# The mechanism here retrieves a random set of indices which pull from the from the input X and y
def getBatch(X,y,batch):
    while True:
        iList= np.random.choice(len(X), size=batch, replace=(len(X)<batch))
        Xbatch= X[iList,:,:,:]
        ybatch= y[iList]
        yield Xbatch, ybatch

# Define the training routine
# Input is a global stream of features and labels
def trainModel():
    global X_train, y_train
    X_train= np.array(X_train)
    y_train= np.array(y_train)
    
    X_train= np.reshape(X_train, (-1, 18,80,1))
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=.01, random_state=21)
    print(X_train.shape, X_valid.shape)
    chek= ModelCheckpoint("model.h5", monitor='val_mean_squared_error', verbose=1, save_best_only=True, mode='min')
    halt= EarlyStopping(monitor='val_mean_squared_error', min_delta=0.0001, patience=50, verbose=1, mode='min')
    # the non-generator version...
    #hist= model.fit(X_train, y_train, callbacks=[chek, halt], validation_split=0.01, batch_size=186, nb_epoch=200, verbose=1) 
    hist= model.fit_generator(getBatch(X_train,y_train,186), samples_per_epoch=8742, nb_epoch=200, callbacks=[chek, halt], validation_data=getBatch(X_valid,y_valid,87),nb_val_samples=87, verbose=1)
    
    # save the model and weights
    deleteFile('model.json')
    deleteFile('model.h5')
    jsonString = model.to_json()
    with open('model.json', 'w') as outfile:
        json.dump(jsonString, outfile)
    model.save_weights('model.h5')
    
    # free up some space
    del X_train
    del y_train

# Define a quick test routine for post-training
def runQuicktest():
    global X_train, y_train
    X_train= []
    y_train= []
    loadQuicktest() #load the three validation images
    re= X_train[0]
    re2= np.reshape(re, (-1, 18,80,1))
    print('Predict:', float(model.predict(re2, batch_size=1)), ' True:', y_train[0])
    re= X_train[1]
    re2= np.reshape(re, (-1, 18,80,1))
    print('Predict:', float(model.predict(re2, batch_size=1)), ' True:', y_train[1])
    re= X_train[2]
    re2= np.reshape(re, (-1, 18,80,1))
    print('Predict:', float(model.predict(re2, batch_size=1)), ' True:', y_train[2])

# Executive
loadData('U') 
#loadQuicktest()  # alternatively use this to preload 3 images for model feasibility
model= createModel()
trainModel()
runQuicktest()

