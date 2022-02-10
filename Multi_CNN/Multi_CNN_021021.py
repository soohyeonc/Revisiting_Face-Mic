import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import PIL
import tensorflow as tf
import functools
import pathlib
import scipy.stats as stats
import cv2
import random

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from tensorflow.keras.optimizers import Adam

from pickle import load
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from keras.models import load_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import BatchNormalization
from keras.layers import Rescaling
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.merge import concatenate

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.image as mpimg

#%matplotlib inline




def Multi_CNN(image_path, ran_state):

    #####################################################################################################################################################################################################################################################
    batch_size = 32
    img_height = 270
    img_width = 120
    #img_folder= r'Segmentedspectrograms'

    def create_dataset(img_folder):
        img_data_array=[]
        class_name=[]

        for dir1 in os.listdir(img_folder):
            for file in os.listdir(os.path.join(img_folder, dir1)):
                image_path= os.path.join(img_folder, dir1, file)
                image= cv2.imread(image_path, cv2.COLOR_BGR2RGB)
                image= cv2.resize(image, (img_width, img_height),interpolation = cv2.INTER_AREA)
                image=np.array(image)
                image = image.astype('float64')
                image /= 255
                img_data_array.append(image)
                class_name.append(dir1)
        return img_data_array, class_name
    # extract the image array and class name



    img_data, class_name =create_dataset(image_path+'/train')

    train_target_dict={k: v for v, k in enumerate(np.unique(class_name))}
    #print(target_dict)

    train_target_val=  [train_target_dict[class_name[i]] for i in range(len(class_name))]
    #print(target_val)

    train_x = np.array(img_data, np.float64)
    train_y = np.array(list(map(int, train_target_val)), np.float64)
    #print(val_ds)


    ####### image dataset

    test_img_data, test_class_name =create_dataset(image_path+'/test')

    test_target_dict={k: v for v, k in enumerate(np.unique(test_class_name))}
    #print(target_dict)

    test_target_val=  [test_target_dict[test_class_name[i]] for i in range(len(test_class_name))]
    #print(target_val)

    test_x = np.array(test_img_data, np.float64)
    test_y =np.array(list(map(int, test_target_val)), np.float64)


    #history = model.fit(x=np.array(img_data, np.float32), y=np.array(list(map(int,target_val)), np.float32), epochs=5)

    ###################################################################################################################################################################################################################################################################

    Fs = 108
    frame_size = Fs*5
    hop_size = Fs*2

    def get_frames(df, frame_size, hop_size, c):

        N_FEATURES = 1

        frames = []
        labels = []
        for i in range(0, len(df) - frame_size, hop_size):
            if(c == 'x'):
                x = df['x'].values[i: i + frame_size]
            elif(c == 'y'):
                x = df['y'].values[i: i + frame_size]
            elif(c == 'z'):
                x = df['z'].values[i: i + frame_size]
            else:
                print("Error")
                break

            # Retrieve the most often used label in this segment
            label = stats.mode(df['label'][i: i + frame_size])[0][0]
            frames.append([x])
            labels.append(label)

        # Bring the segments into a better shape
        frames = np.asarray(frames).reshape(-1, frame_size, N_FEATURES)
        labels = np.asarray(labels)

        return frames, labels

    ####################################################################################################33

    merged = pd.read_csv("converted/merged_x.csv")

    merged['x'] = merged['x'].astype('float64')

    AS_M_1 = merged[merged['ID']=='AS_M_1'].head(13017).copy()
    AS_M_2 = merged[merged['ID']=='AS_M_2'].head(13017).copy()
    ME_F_1 = merged[merged['ID']=='ME_F_1'].head(13017).copy()
    ME_M_1 = merged[merged['ID']=='ME_M_1'].head(13017).copy()
    ME_M_2 = merged[merged['ID']=='ME_M_2'].head(13017).copy()
    ME_M_3 = merged[merged['ID']=='ME_M_3'].head(13017).copy()
    ME_M_4 = merged[merged['ID']=='ME_M_4'].head(13017).copy()
    ME_M_5 = merged[merged['ID']=='ME_M_5'].head(13017).copy()

    balanced_data = pd.DataFrame()
    balanced_data = balanced_data.append([AS_M_1, AS_M_2, ME_F_1, ME_M_1, ME_M_2, ME_M_3,ME_M_4,ME_M_5])

    label = LabelEncoder()
    balanced_data['label'] = label.fit_transform(balanced_data['ID'])

    X = balanced_data[['x']]
    y = balanced_data['label']


    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    scaled_X = pd.DataFrame(data = X, columns = ['x'])
    scaled_X['label'] = y.values


    X, y = get_frames(scaled_X, frame_size, hop_size, 'x')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = ran_state, stratify = y)

    X_train = X_train.reshape(384, 540, 1, 1)
    X_test = X_test.reshape(96, 540, 1, 1)

    #########################################################################################################


    merged_y = pd.read_csv("converted/merged_y.csv")

    merged_y['y'] = merged_y['y'].astype('float64')

    AS_M_1_y = merged_y[merged_y['ID']=='AS_M_1'].head(13017).copy()
    AS_M_2_y = merged_y[merged_y['ID']=='AS_M_2'].head(13017).copy()
    ME_F_1_y = merged_y[merged_y['ID']=='ME_F_1'].head(13017).copy()
    ME_M_1_y = merged_y[merged_y['ID']=='ME_M_1'].head(13017).copy()
    ME_M_2_y = merged_y[merged_y['ID']=='ME_M_2'].head(13017).copy()
    ME_M_3_y = merged_y[merged_y['ID']=='ME_M_3'].head(13017).copy()
    ME_M_4_y = merged_y[merged_y['ID']=='ME_M_4'].head(13017).copy()
    ME_M_5_y = merged_y[merged_y['ID']=='ME_M_5'].head(13017).copy()

    balanced_data_y = pd.DataFrame()
    balanced_data_y = balanced_data_y.append([AS_M_1_y, AS_M_2_y, ME_F_1_y, ME_M_1_y, ME_M_2_y, ME_M_3_y, ME_M_4_y, ME_M_5_y])

    label_y = LabelEncoder()
    balanced_data_y['label'] = label_y.fit_transform(balanced_data_y['ID'])

    X_y = balanced_data_y[['y']]
    y_y = balanced_data_y['label']


    scaler_y = StandardScaler()
    X_y = scaler_y.fit_transform(X_y)

    scaled_X_y = pd.DataFrame(data = X_y, columns = ['y'])
    scaled_X_y['label'] = y_y.values


    X_y, y_y = get_frames(scaled_X_y, frame_size, hop_size, 'y')

    X_train_y, X_test_y, y_train_y, y_test_y = train_test_split(X_y, y_y, test_size = 0.2, random_state = ran_state, stratify = y)

    X_train_y = X_train_y.reshape(384, 540, 1, 1)
    X_test_y = X_test_y.reshape(96, 540, 1, 1)

    #########################################################################################################



    merged_z = pd.read_csv("converted/merged_z.csv")

    merged_z['z'] = merged_z['z'].astype('float64')

    AS_M_1_z = merged_z[merged_z['ID']=='AS_M_1'].head(13017).copy()
    AS_M_2_z = merged_z[merged_z['ID']=='AS_M_2'].head(13017).copy()
    ME_F_1_z = merged_z[merged_z['ID']=='ME_F_1'].head(13017).copy()
    ME_M_1_z = merged_z[merged_z['ID']=='ME_M_1'].head(13017).copy()
    ME_M_2_z = merged_z[merged_z['ID']=='ME_M_2'].head(13017).copy()
    ME_M_3_z = merged_z[merged_z['ID']=='ME_M_3'].head(13017).copy()
    ME_M_4_z = merged_z[merged_z['ID']=='ME_M_4'].head(13017).copy()
    ME_M_5_z = merged_z[merged_z['ID']=='ME_M_5'].head(13017).copy()

    balanced_data_z = pd.DataFrame()
    balanced_data_z = balanced_data_z.append([AS_M_1_z, AS_M_2_z, ME_F_1_z, ME_M_1_z, ME_M_2_z, ME_M_3_z, ME_M_4_z, ME_M_5_z])

    label_z = LabelEncoder()
    balanced_data_z['label'] = label_z.fit_transform(balanced_data_z['ID'])

    X_z = balanced_data_z[['z']]
    y_z = balanced_data_z['label']


    scaler_z = StandardScaler()
    X_z = scaler_z.fit_transform(X_z)

    scaled_X_z = pd.DataFrame(data = X_z, columns = ['z'])
    scaled_X_z['label'] = y_z.values


    X_z, y_z = get_frames(scaled_X_z, frame_size, hop_size, 'z')

    X_train_z, X_test_z, y_train_z, y_test_z = train_test_split(X_z, y_z, test_size = 0.2, random_state = ran_state, stratify = y)

    X_train_z = X_train_z.reshape(384, 540, 1, 1)
    X_test_z = X_test_z.reshape(96, 540, 1, 1)

    #########################################################################################################




    # define the model
    def define_model():

      # channel 1
      inputs1 = Input(shape=(270, 120, 3))
      #data_augmentation = keras.Sequential([layers.RandomFlip("horizontal",input_shape=(270, 120, 3)), layers.RandomRotation(0.1), layers.RandomZoom(0.1),])(inputs1)
      first_batch = BatchNormalization()(inputs1)
      #first_batch = BatchNormalization()(data_augmentation)
      first_conv2 = Conv2D(32, 2, padding='same', activation='relu')(first_batch)
      first_pool2 = MaxPooling2D()(first_conv2)
      first_conv2_2 = Conv2D(64, 2, padding='same', activation='relu')(first_pool2)
      first_pool2_2 = MaxPooling2D()(first_conv2_2)
      #first_conv2_3 = Conv2D(128, 2, padding='same', activation='relu')(first_pool2_2)
      #first_pool2_3 = MaxPooling2D()(first_conv2_3)
      first_flat1 = Flatten()(first_pool2_2)
      first_dense1 = Dense(128, activation='relu')(first_flat1)
      first_drop1 = Dropout(0.7)(first_dense1)
      #first_dense2 = (8, activation='softmax')(first_drop1)

      # channel 2
      inputs2 = Input(shape=(540, 1, 1))
      #embedding2 = Embedding(vocab_size, 100)(inputs2)
      second_batch = BatchNormalization()(inputs2)
      second_conv2 = Conv2D(32, (2,2), padding="same", activation='relu')(second_batch)
      second_conv2_2 = Conv2D(64, (2,2), padding="same", activation='relu')(second_conv2)
      second_conv2_3 = Conv2D(128, (2,2), padding="same", activation='relu')(second_conv2_2)
      second_flat1 = Flatten()(second_conv2_3)
      second_dense1 = Dense(128, activation='relu')(second_flat1)
      second_drop1 = Dropout(0.7)(second_dense1)
      #second_dense2 = (8, activation='softmax')(second_drop1)

      # channel 3
      inputs3 = Input(shape=(540, 1, 1))
      #embedding2 = Embedding(vocab_size, 100)(inputs2)
      third_batch = BatchNormalization()(inputs3)
      third_conv2 = Conv2D(32, (2,2), padding="same", activation='relu')(third_batch)
      third_conv2_2 = Conv2D(64, (2,2), padding="same", activation='relu')(third_conv2)
      third_onv2_3 = Conv2D(128, (2,2), padding="same", activation='relu')(third_conv2_2)
      third_flat1 = Flatten()(third_onv2_3)
      third_dense1 = Dense(128, activation='relu')(third_flat1)
      third_drop1 = Dropout(0.7)(third_dense1)
      #second_dense2 = (8, activation='softmax')(second_drop1)

      # channel 4
      inputs4 = Input(shape=(540, 1, 1))
      #embedding2 = Embedding(vocab_size, 100)(inputs2)
      forth_batch = BatchNormalization()(inputs4)
      forth_conv2 = Conv2D(32, (2,2), padding="same", activation='relu')(forth_batch)
      forth_conv2_2 = Conv2D(64, (2,2), padding="same", activation='relu')(forth_conv2)
      forth_conv2_3 = Conv2D(128, (2,2), padding="same", activation='relu')(forth_conv2_2)
      forth_flat1 = Flatten()(forth_conv2_3)
      forth_dense1 = Dense(128, activation='relu')(forth_flat1)
      forth_drop1 = Dropout(0.7)(forth_dense1)
      #second_dense2 = (8, activation='softmax')(second_drop1)


      # merge
      #no dropout
      #merged = concatenate([first_dense1, second_dense1, third_dense1, forth_dense1])

      #dropout
      merged = concatenate([first_drop1, second_drop1, third_drop1, forth_drop1])

      # interpretation
      flat1 = Flatten()(merged)
      #dense1 = Dense(256, activation='relu')(flat1)
      dense2 = Dense(128, activation='relu')(flat1)
      #dense3 = Dense(64, activation='relu')(dense2)
      dense4 = Dense(32, activation='relu')(dense2)
      #dense5 = Dense(16, activation='relu')(dense4)

      outputs = Dense(8, activation='softmax')(dense4)
      model = Model(inputs=[inputs1, inputs2, inputs3, inputs4], outputs=outputs)
      # compile
      model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
      #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

      # summarize
      print(model.summary())
      plot_model(model, show_shapes=True, to_file='multichannel.png')
      return model


    model = define_model()
    ep = 50 #epoch

    history = model.fit([train_x, X_train, X_train_y, X_train_z], train_y,  epochs=ep)
    model.save('model.h5')

    def plot_learningCurve(history, epochs):
      # Plot training & validation accuracy values
      epoch_range = range(1, epochs+1)
      plt.plot(epoch_range, history.history['accuracy'])
      #plt.plot(epoch_range, history.history['val_accuracy'])
      plt.title('Model Training Accuracy')
      plt.ylabel('Accuracy')
      plt.xlabel('Epoch')
      #plt.legend(['Train', 'Val'], loc='upper left')
      plt.legend('Train', loc='upper left')
      plt.show()

      # Plot training & validation loss values
      plt.plot(epoch_range, history.history['loss'])
      #plt.plot(epoch_range, history.history['val_loss'])
      plt.title('Model Training Loss')
      plt.ylabel('Loss')
      plt.xlabel('Epoch')
      #plt.legend(['Train', 'Val'], loc='upper left')
      plt.legend('Train', loc='upper left')
      plt.show()


    #plot_learningCurve(history, ep)

    print()
    #print("Evaludation")
    # load the model
    model = load_model('model.h5')

    # evaluate model on training dataset
    loss1, acc1 = model.evaluate([train_x, X_train, X_train_y, X_train_z], train_y, verbose=0)
    print('Train Accuracy: %f' % (acc1*100))
    print('Train Loss: %f' % (loss1*100))

    # evaluate model on test dataset dataset
    loss2, acc2 = model.evaluate([test_x, X_test, X_test_y, X_test_z], test_y, verbose=0)
    print('Test Accuracy: %f' % (acc2*100))
    print('Test Loss: %f' % (loss2*100))

    return loss1, acc1, loss2, acc2



path1 ='Segmentedspectrograms_1'
path2 ='Segmentedspectrograms_2'
path3 ='Segmentedspectrograms_3'
path4 ='Segmentedspectrograms_4'
path5 ='Segmentedspectrograms_5'

loss1_1, acc1_1, loss2_1, acc2_1 = Multi_CNN(path1, 40)
loss1_2, acc1_2, loss2_2, acc2_2 = Multi_CNN(path2, 41)
loss1_3, acc1_3, loss2_3, acc2_3 = Multi_CNN(path3, 42)
loss1_4, acc1_4, loss2_4, acc2_4 = Multi_CNN(path4, 43)
loss1_5, acc1_5, loss2_5, acc2_5 = Multi_CNN(path5, 44)


print()
print("Evaluation")
print("====== First data set")
print('Train Accuracy: %f' % (acc1_1*100))
print('Train Loss: %f' % (loss1_1*100))
print('Test Accuracy: %f' % (acc2_1*100))
print('Test Loss: %f' % (loss2_1*100))
print()

print("====== Second data set")
print('Train Accuracy: %f' % (acc1_2*100))
print('Train Loss: %f' % (loss1_2*100))
print('Test Accuracy: %f' % (acc2_2*100))
print('Test Loss: %f' % (loss2_2*100))
print()

print("====== Third data set")
print('Train Accuracy: %f' % (acc1_3*100))
print('Train Loss: %f' % (loss1_3*100))
print('Test Accuracy: %f' % (acc2_3*100))
print('Test Loss: %f' % (loss2_3*100))
print()

print("====== Forth data set")
print('Train Accuracy: %f' % (acc1_4*100))
print('Train Loss: %f' % (loss1_4*100))
print('Test Accuracy: %f' % (acc2_4*100))
print('Test Loss: %f' % (loss2_4*100))
print()

print("====== Fifth data set")
print('Train Accuracy: %f' % (acc1_5*100))
print('Train Loss: %f' % (loss1_5*100))
print('Test Accuracy: %f' % (acc2_5*100))
print('Test Loss: %f' % (loss2_5*100))
