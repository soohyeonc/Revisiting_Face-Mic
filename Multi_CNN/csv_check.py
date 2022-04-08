import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import LabelEncoder
import scipy.stats as stats


train_set = pd.read_csv("data/csvfiles/test/data_4/merged_train.csv")

test_set = pd.read_csv("data/csvfiles/test/data_4/merged_test.csv")


#print(first_train_x.info())

train_set_ID = train_set['ID'].value_counts()
print(train_set_ID)

test_set_ID = test_set['ID'].value_counts()
print(test_set_ID)

train_number = 1750
test_number = 600

# train_number = 3700
# test_number = 1250

AS_F_1 = train_set[train_set['ID']=='AS_F_1'].head(train_number).copy()
AS_F_3 = train_set[train_set['ID']=='AS_F_3'].head(train_number).copy()
AS_F_4 = train_set[train_set['ID']=='AS_F_4'].head(train_number).copy()
AS_F_5 = train_set[train_set['ID']=='AS_F_6'].head(train_number).copy()

AS_M_1 = train_set[train_set['ID']=='AS_M_1'].head(train_number).copy()
AS_M_2 = train_set[train_set['ID']=='AS_M_2'].head(train_number).copy()
AS_M_3 = train_set[train_set['ID']=='AS_M_3'].head(train_number).copy()
AS_M_4 = train_set[train_set['ID']=='AS_M_4'].head(train_number).copy()

ME_F_1 = train_set[train_set['ID']=='ME_F_1'].head(train_number).copy()
ME_F_4 = train_set[train_set['ID']=='ME_F_4'].head(train_number).copy()
ME_F_5 = train_set[train_set['ID']=='ME_F_5'].head(train_number).copy()

ME_M_4 = train_set[train_set['ID']=='ME_M_4'].head(train_number).copy()
ME_M_7 = train_set[train_set['ID']=='ME_M_7'].head(train_number).copy()
ME_M_8 = train_set[train_set['ID']=='ME_M_8'].head(train_number).copy()


AS_F_1_t = test_set[test_set['ID']=='AS_F_1'].head(test_number).copy()
AS_F_3_t = test_set[test_set['ID']=='AS_F_3'].head(test_number).copy()
AS_F_4_t = test_set[test_set['ID']=='AS_F_4'].head(test_number).copy()
AS_F_5_t = test_set[test_set['ID']=='AS_F_6'].head(test_number).copy()

AS_M_1_t = test_set[test_set['ID']=='AS_M_1'].head(test_number).copy()
AS_M_2_t = test_set[test_set['ID']=='AS_M_2'].head(test_number).copy()
AS_M_3_t = test_set[test_set['ID']=='AS_M_3'].head(test_number).copy()
AS_M_4_t = test_set[test_set['ID']=='AS_M_4'].head(test_number).copy()

ME_F_1_t = test_set[test_set['ID']=='ME_F_1'].head(test_number).copy()
ME_F_4_t = test_set[test_set['ID']=='ME_F_4'].head(test_number).copy()
ME_F_5_t = test_set[test_set['ID']=='ME_F_5'].head(test_number).copy()

ME_M_4_t = test_set[test_set['ID']=='ME_M_4'].head(test_number).copy()
ME_M_7_t = test_set[test_set['ID']=='ME_M_7'].head(test_number).copy()
ME_M_8_t = test_set[test_set['ID']=='ME_M_8'].head(test_number).copy()




balanced_data = pd.DataFrame()
balanced_data = balanced_data.append([AS_F_1, AS_F_3, AS_F_4, AS_F_5, AS_M_1, AS_M_2, AS_M_3, AS_M_4, ME_F_1, ME_F_4, ME_F_5, ME_M_4, ME_M_7, ME_M_8])

balanced_data_t = pd.DataFrame()
balanced_data_t = balanced_data_t.append([AS_F_1_t, AS_F_3_t, AS_F_4_t, AS_F_5_t, AS_M_1_t, AS_M_2_t, AS_M_3_t, AS_M_4_t, ME_F_1_t, ME_F_4_t, ME_F_5_t, ME_M_4_t, ME_M_7_t, ME_M_8_t])


label = LabelEncoder()
balanced_data['label'] = label.fit_transform(balanced_data['ID'])
balanced_data

label_t = LabelEncoder()
balanced_data_t['label'] = label_t.fit_transform(balanced_data_t['ID'])
balanced_data_t

X = balanced_data[['Acc_x']]
y = balanced_data['label']

X_t = balanced_data_t[['Acc_x']]
y_t = balanced_data_t['label']

scaler = StandardScaler()
X = scaler.fit_transform(X)

scaled_X = pd.DataFrame(data = X, columns = ['Acc_x'])
scaled_X['label'] = y.values


scaler_t = StandardScaler()
X_t = scaler_t.fit_transform(X_t)

scaled_X_t = pd.DataFrame(data = X_t, columns = ['Acc_x'])
scaled_X_t['label'] = y_t.values

#756, 252
Fs = 32
frame_size = Fs*11 # 80
hop_size = Fs*2 # 40

def get_frames(df, frame_size, hop_size):

    N_FEATURES = 1

    frames = []
    labels = []
    for i in range(0, len(df) - frame_size, hop_size):
        x = df['Acc_x'].values[i: i + frame_size]
        #y = df['y'].values[i: i + frame_size]
        #z = df['z'].values[i: i + frame_size]

        # Retrieve the most often used label in this segment
        label = stats.mode(df['label'][i: i + frame_size])[0][0]
        frames.append([x])
        labels.append(label)

    # Bring the segments into a better shape
    frames = np.asarray(frames).reshape(-1, frame_size, N_FEATURES)
    labels = np.asarray(labels)

    return frames, labels

print("Fs = ", Fs)
print("frame_size = ", frame_size)

X, y = get_frames(scaled_X, frame_size, hop_size)
print(X.shape)
print(y.shape)

X_t, y_t = get_frames(scaled_X_t, frame_size, hop_size)

print(X_t.shape)
print(y_t.shape)

#378
#126
