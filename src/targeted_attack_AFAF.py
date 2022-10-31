import os
import cv2
import PIL
import random
import pathlib
import functools
import numpy as np
import pandas as pd
from numpy import array
from pickle import load
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential

from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Rescaling
from keras.models import load_model
from keras.layers.merge import concatenate
from keras.layers import BatchNormalization
from keras.utils.vis_utils import plot_model
from keras.layers.convolutional import Conv2D
from keras.preprocessing.text import Tokenizer
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder,OneHotEncoder


def Multi_CNN(image_path, csv_path):
############################################################################################################
# spectrograms #############################################################################################
    def create_dataset(img_folder):
        img_height = 270
        img_width = 120

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

#######################
#training spectrograms#
    img_data, class_name =create_dataset("data/spectrograms/"+image_path+'/train')

    train_target_dict={k: v for v, k in enumerate(np.unique(class_name))}

    train_target_val=  [train_target_dict[class_name[i]] for i in range(len(class_name))]

    train_x = np.array(img_data, np.float64)
    train_y = np.array(list(map(int, train_target_val)), np.float64)

######################
#testing spectrograms#
    test_img_data, test_class_name =create_dataset("data/spectrograms/"+image_path+'/test')

    test_target_dict={k: v for v, k in enumerate(np.unique(test_class_name))}

    test_target_val=  [test_target_dict[test_class_name[i]] for i in range(len(test_class_name))]

    test_x = np.array(test_img_data, np.float64)
    test_y = np.array(list(map(int, test_target_val)), np.float64)

############################################################################################################
# csv files ################################################################################################
    def get_frames(df, c):
        Fs = 35
        frame_size = Fs*11
        hop_size = Fs*2
        N_FEATURES = 1
        frames = []
        labels = []

        for i in range(0, len(df) - frame_size, hop_size):
            if(c == 'Acc_x'):
                x = df['Acc_x'].values[i: i + frame_size]
            elif(c == 'Acc_y'):
                x = df['Acc_y'].values[i: i + frame_size]
            elif(c == 'Acc_z'):
                x = df['Acc_z'].values[i: i + frame_size]

            elif(c == 'Velo_x'):
                x = df['Velo_x'].values[i: i + frame_size]
            elif(c == 'Velo_y'):
                x = df['Velo_y'].values[i: i + frame_size]
            elif(c == 'Velo_z'):
                x = df['Velo_z'].values[i: i + frame_size]

            elif(c == 'Pos_x'):
                x = df['Pos_x'].values[i: i + frame_size]
            elif(c == 'Pos_y'):
                x = df['Pos_y'].values[i: i + frame_size]
            elif(c == 'Pos_z'):
                x = df['Pos_z'].values[i: i + frame_size]
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

######################
#training csv#
    def training_set(c, path):
        train_set = pd.read_csv("data/csvfiles/"+path)
        train_set[c] = train_set[c].astype('float64')

        train_number = 5720 # 1:1 # train data : test data
        reshape_number = 567 # 1:1 # train data : test data
        framesize = 385

        AS_F_3 = train_set[train_set['ID']=='AS_F_3'].head(train_number).copy()
        #AS_F_4 = train_set[train_set['ID']=='AS_F_4'].head(train_number).copy()

        AS_M_3 = train_set[train_set['ID']=='AS_M_3'].head(train_number).copy()
        AS_M_4 = train_set[train_set['ID']=='AS_M_4'].head(train_number).copy()

        ME_F_5 = train_set[train_set['ID']=='ME_F_5'].head(train_number).copy()
        ME_F_6 = train_set[train_set['ID']=='ME_F_6'].head(train_number).copy()

        ME_M_7 = train_set[train_set['ID']=='ME_M_7'].head(train_number).copy()
        ME_M_8 = train_set[train_set['ID']=='ME_M_8'].head(train_number).copy()

        balanced_data = pd.DataFrame()
        balanced_data = balanced_data.append([AS_F_3, AS_M_3, AS_M_4, ME_F_5, ME_M_7, ME_M_8, ME_F_6])

        label = LabelEncoder()
        balanced_data['label'] = label.fit_transform(balanced_data['ID'])

        X = balanced_data[[c]]
        y = balanced_data['label']

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        scaled_X = pd.DataFrame(data = X, columns = [c])
        scaled_X['label'] = y.values

        X, y = get_frames(scaled_X, c)
        X = X.reshape(reshape_number, framesize, 1, 1)

        return X, y

######################
#testing csv#
    def testing_set(c, path):
        test_set = pd.read_csv("data/csvfiles/"+path)
        test_set[c] = test_set[c].astype('float64')

        test_number = 1944 # 1:1 # train data : test data
        reshape_number = 189 # 1:1 # train data : test data
        framesize = 385


        AS_F_3_t = test_set[test_set['ID']=='AS_F_3'].head(test_number).copy()
        #AS_F_4_t = test_set[test_set['ID']=='AS_F_4'].head(test_number).copy()

        AS_M_3_t = test_set[test_set['ID']=='AS_M_3'].head(test_number).copy()
        AS_M_4_t = test_set[test_set['ID']=='AS_M_4'].head(test_number).copy()

        ME_F_5_t = test_set[test_set['ID']=='ME_F_5'].head(test_number).copy()
        ME_F_6_t = test_set[test_set['ID']=='ME_F_6'].head(test_number).copy()

        ME_M_7_t = test_set[test_set['ID']=='ME_M_7'].head(test_number).copy()
        ME_M_8_t = test_set[test_set['ID']=='ME_M_8'].head(test_number).copy()


        balanced_data_t = pd.DataFrame()
        balanced_data_t = balanced_data_t.append([AS_F_3_t, AS_M_3_t, AS_M_4_t, ME_F_5_t, ME_M_7_t, ME_M_8_t, ME_F_6_t])

        label_t = LabelEncoder()
        balanced_data_t['label'] = label_t.fit_transform(balanced_data_t['ID'])

        X_t = balanced_data_t[[c]]
        y_t = balanced_data_t['label']

        scaler_t = StandardScaler()
        X_t = scaler_t.fit_transform(X_t)

        scaled_X_t = pd.DataFrame(data = X_t, columns = [c])
        scaled_X_t['label'] = y_t.values

        X_t, y_t = get_frames(scaled_X_t, c)
        X_t = X_t.reshape(reshape_number, framesize, 1, 1) # 1:1  # train data : test data

        return X_t, y_t

###### each training data set ######
    X_x, y_x = training_set('Acc_x', csv_path+"train.csv")
    X_y, y_y = training_set('Acc_y', csv_path+"train.csv")
    X_z, y_z = training_set('Acc_z', csv_path+"train.csv")

    X_x_vel, y_x_vel = training_set('Velo_x', csv_path+"train.csv")
    X_y_vel, y_y_vel = training_set('Velo_y', csv_path+"train.csv")
    X_z_vel, y_z_vel = training_set('Velo_z', csv_path+"train.csv")

    X_x_pos, y_x_pos = training_set('Pos_x', csv_path+"train.csv")
    X_y_pos, y_y_pos = training_set('Pos_y', csv_path+"train.csv")
    X_z_pos, y_z_pos = training_set('Pos_z', csv_path+"train.csv")

###### each test data set ######
    X_t_x, y_t_x = testing_set('Acc_x', csv_path+"test.csv")
    X_t_y, y_t_y = testing_set('Acc_y', csv_path+"test.csv")
    X_t_z, y_t_z = testing_set('Acc_z', csv_path+"test.csv")

    X_t_x_vel, y_t_x_vel = testing_set('Velo_x', csv_path+"test.csv")
    X_t_y_vel, y_t_y_vel = testing_set('Velo_y', csv_path+"test.csv")
    X_t_z_vel, y_t_z_vel = testing_set('Velo_z', csv_path+"test.csv")

    X_t_x_pos, y_t_x_pos = testing_set('Pos_x', csv_path+"test.csv")
    X_t_y_pos, y_t_y_pos = testing_set('Pos_y', csv_path+"test.csv")
    X_t_z_pos, y_t_z_pos = testing_set('Pos_z', csv_path+"test.csv")

############################################################################################################
# define the model #########################################################################################
    def define_model():
        shape_number = 385
        firstpara = 32
        secondpara = 64
        thirdpara = 128
        densepara = 128
        dropoutpara = 0.7

############################################
# spectrograms #############################
        # channel 1
        inputs1 = Input(shape=(270, 120, 3))
        first_batch = BatchNormalization()(inputs1)
        first_conv2 = Conv2D(32, 2, padding='same', activation='relu')(first_batch)
        first_pool2 = MaxPooling2D()(first_conv2)
        first_conv2_2 = Conv2D(64, 2, padding='same', activation='relu')(first_pool2)
        first_pool2_2 = MaxPooling2D()(first_conv2_2)
        first_flat1 = Flatten()(first_pool2_2)
        first_dense1 = Dense(densepara, activation='relu')(first_flat1)
        first_drop1 = Dropout(0.7)(first_dense1)

############################################
# accelrometer #############################
        # channel 1 for acc
        inputs2 = Input(shape=(shape_number, 1, 1))
        second_batch = BatchNormalization()(inputs2)
        second_conv2 = Conv2D(firstpara, (2,2), padding="same", activation='relu')(second_batch)
        second_conv2_2 = Conv2D(secondpara, (2,2), padding="same", activation='relu')(second_conv2)
        second_conv2_3 = Conv2D(thirdpara, (2,2), padding="same", activation='relu')(second_conv2_2)
        second_flat1 = Flatten()(second_conv2_3)
        second_dense1 = Dense(densepara, activation='relu')(second_flat1)
        second_drop1 = Dropout(dropoutpara)(second_dense1)

        # channel 2 for acc
        inputs3 = Input(shape=(shape_number, 1, 1))
        third_batch = BatchNormalization()(inputs3)
        third_conv2 = Conv2D(firstpara, (2,2), padding="same", activation='relu')(third_batch)
        third_conv2_2 = Conv2D(secondpara, (2,2), padding="same", activation='relu')(third_conv2)
        third_conv2_3 = Conv2D(thirdpara, (2,2), padding="same", activation='relu')(third_conv2_2)
        third_flat1 = Flatten()(third_conv2_3)
        third_dense1 = Dense(densepara, activation='relu')(third_flat1)
        third_drop1 = Dropout(dropoutpara)(third_dense1)

        # channel 3 for acc
        inputs4 = Input(shape=(shape_number, 1, 1))
        forth_batch = BatchNormalization()(inputs4)
        forth_conv2 = Conv2D(firstpara, (2,2), padding="same", activation='relu')(forth_batch)
        forth_conv2_2 = Conv2D(secondpara, (2,2), padding="same", activation='relu')(forth_conv2)
        forth_conv2_3 = Conv2D(thirdpara, (2,2), padding="same", activation='relu')(forth_conv2_2)
        forth_flat1 = Flatten()(forth_conv2_3)
        forth_dense1 = Dense(densepara, activation='relu')(forth_flat1)
        forth_drop1 = Dropout(dropoutpara)(forth_dense1)

############################################
# 3Dspeed ##################################
        # channel 1 for vel
        inputs5 = Input(shape=(shape_number, 1, 1))
        vel_second_batch = BatchNormalization()(inputs5)
        vel_second_conv2 = Conv2D(firstpara, (2,2), padding="same", activation='relu')(vel_second_batch)
        vel_second_conv2_2 = Conv2D(secondpara, (2,2), padding="same", activation='relu')(vel_second_conv2)
        vel_second_conv2_3 = Conv2D(thirdpara, (2,2), padding="same", activation='relu')(vel_second_conv2_2)
        vel_second_flat1 = Flatten()(vel_second_conv2_3)
        vel_second_dense1 = Dense(densepara, activation='relu')(vel_second_flat1)
        vel_second_drop1 = Dropout(dropoutpara)(vel_second_dense1)

        # channel 2 for vel
        inputs6 = Input(shape=(shape_number, 1, 1))
        vel_third_batch = BatchNormalization()(inputs6)
        vel_third_conv2 = Conv2D(firstpara, (2,2), padding="same", activation='relu')(vel_third_batch)
        vel_third_conv2_2 = Conv2D(secondpara, (2,2), padding="same", activation='relu')(vel_third_conv2)
        vel_third_conv2_3 = Conv2D(thirdpara, (2,2), padding="same", activation='relu')(vel_third_conv2_2)
        vel_third_flat1 = Flatten()(vel_third_conv2_3)
        vel_third_dense1 = Dense(densepara, activation='relu')(vel_third_flat1)
        vel_third_drop1 = Dropout(dropoutpara)(vel_third_dense1)

        # channel 3 for vel
        inputs7 = Input(shape=(shape_number, 1, 1))
        vel_forth_batch = BatchNormalization()(inputs7)
        vel_forth_conv2 = Conv2D(firstpara, (2,2), padding="same", activation='relu')(vel_forth_batch)
        vel_forth_conv2_2 = Conv2D(secondpara, (2,2), padding="same", activation='relu')(vel_forth_conv2)
        vel_forth_conv2_3 = Conv2D(thirdpara, (2,2), padding="same", activation='relu')(vel_forth_conv2_2)
        vel_forth_flat1 = Flatten()(vel_forth_conv2_3)
        vel_forth_dense1 = Dense(densepara, activation='relu')(vel_forth_flat1)
        vel_forth_drop1 = Dropout(dropoutpara)(vel_forth_dense1)

############################################
# displacment ##############################
        # channel 1 for pos
        inputs8 = Input(shape=(shape_number, 1, 1))
        pos_second_batch = BatchNormalization()(inputs8)
        pos_second_conv2 = Conv2D(firstpara, (2,2), padding="same", activation='relu')(pos_second_batch)
        pos_second_conv2_2 = Conv2D(secondpara, (2,2), padding="same", activation='relu')(pos_second_conv2)
        pos_second_conv2_3 = Conv2D(thirdpara, (2,2), padding="same", activation='relu')(pos_second_conv2_2)
        pos_second_flat1 = Flatten()(pos_second_conv2_3)
        pos_second_dense1 = Dense(densepara, activation='relu')(pos_second_flat1)
        pos_second_drop1 = Dropout(dropoutpara)(pos_second_dense1)

        # channel 2 for pos
        inputs9 = Input(shape=(shape_number, 1, 1))
        pos_third_batch = BatchNormalization()(inputs9)
        pos_third_conv2 = Conv2D(firstpara, (2,2), padding="same", activation='relu')(pos_third_batch)
        pos_third_conv2_2 = Conv2D(secondpara, (2,2), padding="same", activation='relu')(pos_third_conv2)
        pos_third_conv2_3 = Conv2D(thirdpara, (2,2), padding="same", activation='relu')(pos_third_conv2_2)
        pos_third_flat1 = Flatten()(pos_third_conv2_3)
        pos_third_dense1 = Dense(densepara, activation='relu')(pos_third_flat1)
        pos_third_drop1 = Dropout(dropoutpara)(pos_third_dense1)

        # channel 3 for pos
        inputs10 = Input(shape=(shape_number, 1, 1))
        pos_forth_batch = BatchNormalization()(inputs10)
        pos_forth_conv2 = Conv2D(firstpara, (2,2), padding="same", activation='relu')(pos_forth_batch)
        pos_forth_conv2_2 = Conv2D(secondpara, (2,2), padding="same", activation='relu')(pos_forth_conv2)
        pos_forth_conv2_3 = Conv2D(thirdpara, (2,2), padding="same", activation='relu')(pos_forth_conv2_2)
        pos_forth_flat1 = Flatten()(pos_forth_conv2_3)
        pos_forth_dense1 = Dense(densepara, activation='relu')(pos_forth_flat1)
        pos_forth_drop1 = Dropout(dropoutpara)(pos_forth_dense1)

############################################
# concatenate ##############################
        merged = concatenate([first_drop1, second_drop1, third_drop1, forth_drop1, vel_second_drop1, vel_third_drop1, vel_forth_drop1, pos_second_drop1, pos_third_drop1, pos_forth_drop1])

        # interpretation
        flat1 = Flatten()(merged)
        dense1 = Dense(512, activation='relu')(flat1)
        dense2 = Dense(128, activation='relu')(dense1)
        outputs = Dense(7, activation='softmax')(dense2)

        model = Model(inputs=[inputs1, inputs2, inputs3, inputs4, inputs5, inputs6, inputs7, inputs8, inputs9, inputs10], outputs=outputs)
        # compile
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # summarize
        print(model.summary())
        plot_model(model, show_shapes=True, to_file='multichannel.png')
        return model

############################################
# define model #############################
    model = define_model()
    ep = 50 #epoch
    bs = 8 #bath size

    history = model.fit([train_x, X_x, X_y, X_z, X_x_vel, X_y_vel, X_z_vel, X_x_pos, X_y_pos, X_z_pos], train_y, epochs=ep, batch_size = bs)
    model.save('model.h5')

############################################
# ploting model ############################
    def plot_learningCurve(history, epochs):
      # Plot training & validation accuracy values
      epoch_range = range(1, epochs+1)
      plt.plot(epoch_range, history.history['accuracy'])
      plt.title('Model Training Accuracy')
      plt.ylabel('Accuracy')
      plt.xlabel('Epoch')
      plt.legend('Train', loc='upper left')
      plt.show()

      # Plot training & validation loss values
      plt.plot(epoch_range, history.history['loss'])
      plt.title('Model Training Loss')
      plt.ylabel('Loss')
      plt.xlabel('Epoch')
      plt.legend('Train', loc='upper left')
      plt.show()

    #plot_learningCurve(history, ep)
    print()

    # load the model
    model = load_model('model.h5')

    # evaluate model on training dataset
    loss1, acc1 = model.evaluate([train_x, X_x, X_y, X_z, X_x_vel, X_y_vel, X_z_vel, X_x_pos, X_y_pos, X_z_pos], train_y, verbose=0)
    print('Train Accuracy: %f' % (acc1*100))
    print('Train Loss: %f' % (loss1*100))

    # evaluate model on test dataset dataset
    loss2, acc2 = model.evaluate([test_x, X_t_x, X_t_y, X_t_z, X_t_x_vel, X_t_y_vel, X_t_z_vel, X_t_x_pos, X_t_y_pos, X_t_z_pos], test_y, verbose=0)
    print('Test Accuracy: %f' % (acc2*100))
    print('Test Loss: %f' % (loss2*100))

    return loss1, acc1, loss2, acc2

#########################################################################

def Multi_CNN_attack(image_path, csv_path):
############################################################################################################
# spectrograms #############################################################################################
    def create_dataset(img_folder):
        img_height = 270
        img_width = 120

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

######################
#testing spectrograms#
    test_img_data, test_class_name =create_dataset("data/spectrograms/"+image_path+'/test')

    test_target_dict={k: v for v, k in enumerate(np.unique(test_class_name))}

    test_target_val=  [test_target_dict[test_class_name[i]] for i in range(len(test_class_name))]

    test_x = np.array(test_img_data, np.float64)
    test_y = np.array(list(map(int, test_target_val)), np.float64)


############################################################################################################
# csv files ################################################################################################
    def get_frames(df, c):
        Fs = 35
        frame_size = Fs*11
        hop_size = Fs*2
        N_FEATURES = 1
        frames = []
        labels = []

        for i in range(0, len(df) - frame_size, hop_size):
            if(c == 'Acc_x'):
                x = df['Acc_x'].values[i: i + frame_size]
            elif(c == 'Acc_y'):
                x = df['Acc_y'].values[i: i + frame_size]
            elif(c == 'Acc_z'):
                x = df['Acc_z'].values[i: i + frame_size]

            elif(c == 'Velo_x'):
                x = df['Velo_x'].values[i: i + frame_size]
            elif(c == 'Velo_y'):
                x = df['Velo_y'].values[i: i + frame_size]
            elif(c == 'Velo_z'):
                x = df['Velo_z'].values[i: i + frame_size]

            elif(c == 'Pos_x'):
                x = df['Pos_x'].values[i: i + frame_size]
            elif(c == 'Pos_y'):
                x = df['Pos_y'].values[i: i + frame_size]
            elif(c == 'Pos_z'):
                x = df['Pos_z'].values[i: i + frame_size]
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

######################
#testing csv#
    def testing_set(c, path):
        test_set = pd.read_csv("data/csvfiles/"+path)
        test_set[c] = test_set[c].astype('float64')

        test_number = 1944 # 1:1 # train data : test data
        reshape_number = 189 # 1:1 # train data : test data
        framesize = 385


        AS_F_3_t = test_set[test_set['ID']=='AS_F_3'].head(test_number).copy()
        #AS_F_4_t = test_set[test_set['ID']=='AS_F_4'].head(test_number).copy()

        AS_M_3_t = test_set[test_set['ID']=='AS_M_3'].head(test_number).copy()
        AS_M_4_t = test_set[test_set['ID']=='AS_M_4'].head(test_number).copy()

        ME_F_5_t = test_set[test_set['ID']=='ME_F_5'].head(test_number).copy()
        ME_F_6_t = test_set[test_set['ID']=='ME_F_6'].head(test_number).copy()

        ME_M_7_t = test_set[test_set['ID']=='ME_M_7'].head(test_number).copy()
        ME_M_8_t = test_set[test_set['ID']=='ME_M_8'].head(test_number).copy()


        balanced_data_t = pd.DataFrame()
        balanced_data_t = balanced_data_t.append([AS_F_3_t, AS_M_3_t, AS_M_4_t, ME_F_5_t, ME_M_7_t, ME_M_8_t, ME_F_6_t])

        label_t = LabelEncoder()
        balanced_data_t['label'] = label_t.fit_transform(balanced_data_t['ID'])

        X_t = balanced_data_t[[c]]
        y_t = balanced_data_t['label']

        scaler_t = StandardScaler()
        X_t = scaler_t.fit_transform(X_t)

        scaled_X_t = pd.DataFrame(data = X_t, columns = [c])
        scaled_X_t['label'] = y_t.values

        X_t, y_t = get_frames(scaled_X_t, c)
        X_t = X_t.reshape(reshape_number, framesize, 1, 1) # 1:1  # train data : test data

        return X_t, y_t

###### each test data set ######
    X_t_x, y_t_x = testing_set('Acc_x', csv_path+"test.csv")
    X_t_y, y_t_y = testing_set('Acc_y', csv_path+"test.csv")
    X_t_z, y_t_z = testing_set('Acc_z', csv_path+"test.csv")

    X_t_x_vel, y_t_x_vel = testing_set('Velo_x', csv_path+"test.csv")
    X_t_y_vel, y_t_y_vel = testing_set('Velo_y', csv_path+"test.csv")
    X_t_z_vel, y_t_z_vel = testing_set('Velo_z', csv_path+"test.csv")

    X_t_x_pos, y_t_x_pos = testing_set('Pos_x', csv_path+"test.csv")
    X_t_y_pos, y_t_y_pos = testing_set('Pos_y', csv_path+"test.csv")
    X_t_z_pos, y_t_z_pos = testing_set('Pos_z', csv_path+"test.csv")


    # load the model
    model = load_model('model.h5')

    # evaluate model on test dataset dataset
    loss2, acc2 = model.evaluate([test_x, X_t_x, X_t_y, X_t_z, X_t_x_vel, X_t_y_vel, X_t_z_vel, X_t_x_pos, X_t_y_pos, X_t_z_pos], test_y, verbose=0)
    #print('Test Accuracy: %f' % (acc2*100))
    #print('Test Loss: %f' % (loss2*100))

    return loss2, acc2




#######################################################################################################
# print result ########################################################################################
def printing_result(t,loss1_1, acc1_1, loss2_1, acc2_1):
    print()
    print("Evaluation")
    print("===== " + t + " data set")
    print('Train Accuracy: %f' % (acc1_1*100))
    print('Train Loss: %f' % (loss1_1*100))
    print('Test Accuracy: %f' % (acc2_1*100))
    print('Test Loss: %f' % (loss2_1*100))
    print()

#######################################################################################################
# print result ########################################################################################
def printing_attack_result(t, loss2_1, acc2_1):
    print()
    print("Evaluation")
    print("===== " + t + " data set")
    print('Test Accuracy: %f' % (acc2_1*100))
    print('Test Loss: %f' % (loss2_1*100))
    print()

#######################################################################################################
# save target result ##################################################################################
def target_result(t,loss1_1, acc1_1, loss2_1, acc2_1, attack_number):
    file = open("Result/Result_attack"+str(attack_number)+"_target.txt","a")
    file.write('\n')
    file.write("Evaluation \n")
    file.write("====== "+ t +" data set \n")
    file.write('Train Accuracy: ')
    file.write(str(acc1_1*100))
    file.write('\n')
    file.write('Train Loss: ')
    file.write(str(loss1_1*100))
    file.write('\n')
    file.write('Test Accuracy: ')
    file.write(str(acc2_1*100))
    file.write('\n')
    file.write('Test Loss: ')
    file.write(str(loss2_1*100))
    file.write('\n')

#######################################################################################################
# save attack result ##################################################################################
def attack_result(t, loss2_1, acc2_1, attack_number):
    file = open("Result/Result_attack"+str(attack_number)+"_attack.txt","a")
    file.write('\n')
    file.write("Evaluation \n")
    file.write("====== "+ t +" attack data set \n")
    file.write('Test Accuracy: ')
    file.write(str(acc2_1*100))
    file.write('\n')
    file.write('Test Loss: ')
    file.write(str(loss2_1*100))
    file.write('\n')

#######################################################################################################
# add line ###########################################################################################
def addLine(attack_number, f):
    if(f == 0):
        file = open("Result/Result_attack"+str(attack_number)+"_target.txt","a")
    elif(f == 1):
        file = open("Result/Result_attack"+str(attack_number)+"_attack.txt","a")
    file.write('\n')
    file.write('==================================================')


#####################################################################################################
# iteraitions #######################################################################################
attack_number = 4

datasetList = ["First", "Second", "Third", "Forth"]
#datasetList = ["First"]

count = 0

for i in datasetList:
    count = count + 1

    print(i +" data set")
    path ="targeted_attack/targeted_attack"+str(attack_number)+"/Segmentedspectrograms_" + str(count)
    csv_path = "targeted_attack/targeted_attack/data_"+ str(count) +"/merged_" #target

    loss1, acc1, loss2, acc2 = Multi_CNN(path, csv_path)
    printing_result(i,loss1, acc1, loss2, acc2)
    target_result(i,loss1, acc1, loss2, acc2, attack_number)

    path ="targeted_attack/targeted_attack"+str(attack_number)+"_temp/Segmentedspectrograms_" + str(count)
    csv_path = "targeted_attack/targeted_attack"+str(attack_number)+"/data_"+ str(count) +"/merged_" #attack

    loss2, acc2 = Multi_CNN_attack(path, csv_path)
    printing_attack_result(i, loss2, acc2)
    attack_result(i, loss2, acc2, attack_number)

addLine(attack_number, 0)

addLine(attack_number, 1)









#
