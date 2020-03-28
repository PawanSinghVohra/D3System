import numpy as np
import pandas as pd
import os
import random
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
# filenames_list = os.listdir('../input/SMNI_CMI_TRAIN/SMNI_CMI_TRAIN/') ## list of file names in the directory
EEG_data = pd.DataFrame({}) ## create an empty df that will hold data from each file

# for file_name in tqdm(filenames_list):
#     temp_df = pd.read_csv('../input/SMNI_CMI_TRAIN/SMNI_CMI_TRAIN/' + file_name) ## read from the file to df
#     EEG_data = EEG_data.append(temp_df) ## add the file data to the main df
#
# EEG_data = EEG_data.drop(['Unnamed: 0'], axis=1) ## remove the unused column
# EEG_data.loc[EEG_data
#                  ['matching condition'] == 'S2 nomatch,', 'matching condition'] =  'S2 nomatch' ## remove comma sign from stimulus name

directory = os.path.join("C:\\","Users\VOHRA\PycharmProjects\Analyse\input\SMNI_CMI_TRAIN\SMNI_CMI_TRAIN\co2a0000364")
for root,dirs,files in os.walk(directory):
    for file in files:
       if file.endswith(".gz"):

           temp_df = pd.read_csv("C:\\Users\VOHRA\PycharmProjects\Analyse\input\SMNI_CMI_TRAIN\SMNI_CMI_TRAIN\co2a0000364\\"+file,skiprows = 4,nrows=256,delimiter=' ')  ## read from the file to df
           print(temp_df)
           if file[3]=='a':
               temp_df['Group']='drowsy'
           else:
               temp_df['Group']='Non-drowsy'
           EEG_data = EEG_data.append(temp_df)


# EEG_data.rename(columns={" ":"S.no","#":"x","FP1":"sensor","chan":"itration","0":"sensor value"})
EEG_data.columns=["trial","number","Sensor","Sensor value","Group"]
# EEG_data.drop(['number'], axis = 1)
del EEG_data["number"]
# EEG_data.to_excel(r"C:\Users\VOHRA\PycharmProjects\Analyse\input\output.xlsx")   #Export to external csv

print(EEG_data)



print(EEG_data.dtypes)

print("After")
EEG_data['Group'] = pd.Categorical(EEG_data['Group'])
EEG_data['Group'] = EEG_data.Group.cat.codes


print(EEG_data.dtypes)

# print("After")
#
# EEG_data['number'] = pd.Categorical(EEG_data['number'])
# EEG_data['number'] = EEG_data.Group.cat.codes
#
# print(EEG_data.dtypes)

target = EEG_data.pop('Sensor value')
dataset = tf.data.Dataset.from_tensor_slices((EEG_data.values, target.values))
for feat, targ in dataset.take(5):
  print ('Features: {}, Target: {}'.format(feat, targ))

# print(tf.constant(EEG_data['Sensor value']))

train_dataset = dataset.shuffle(len(EEG_data)).batch(1)
#
# plt.imshow(EEG_data[2])
# plt.show()

def get_compiled_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)
  ])
  # model = tf.keras.Sequential([
  #     tf.keras.layers.Dense(128, activation='relu'),
  #     tf.keras.layers.Dense(10, activation='relu'),
  #     tf.keras.layers.Dense(10)
  # ])

  # model.compile(optimizer='adam',
  #               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  #               metrics=['accuracy'])

  model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                metrics=['accuracy'])
  return model

model = get_compiled_model()
train_label=["drowsy","Non-drowsy"]
# model.fit(train_dataset ,epochs=)
