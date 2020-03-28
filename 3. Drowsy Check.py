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

directory = os.path.join("C:\\","Users\VOHRA\PycharmProjects\Analyse\input\SMNI_CMI_TRAIN\SMNI_CMI_TRAIN\ca2a0000364")
for root,dirs,files in os.walk(directory):
    for file in files:
       if file.endswith(".gz"):

           temp_df = pd.read_csv("C:\\Users\VOHRA\PycharmProjects\Analyse\input\SMNI_CMI_TRAIN\SMNI_CMI_TRAIN\co2a0000364\\"+file,skiprows = 4,nrows=256,delimiter=' ',usecols = [3])  ## read from the file to df
           temp_df=temp_df.transpose()
           print(temp_df)

           if file[3]=='a':
               temp_df['Group']=1
           else:
               temp_df['Group']=0
           EEG_data = EEG_data.append(temp_df)


# EEG_data.rename(columns={" ":"S.no","#":"x","FP1":"sensor","chan":"itration","0":"sensor value"})
# EEG_data.columns=["trial","number","Sensor","Sensor value","Group"]
# del EEG_data["number"]

EEG_data=EEG_data
# EEG_data.to_excel(r"C:\Users\VOHRA\PycharmProjects\Analyse\input\output.xlsx")   #Export to external csv

print(EEG_data)

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

features=[]
for i in range(0,255):
    features.append(i)

# features=[0,1,2,3,4,5]
dataset = (
    tf.data.Dataset.from_tensor_slices(
        (
            [tf.cast(EEG_data[features].values, tf.float32)],
            [tf.cast(EEG_data['Group'].values,tf.int8)]
        )
    )
)
for x,y in dataset:
    print(x,y)
# print("************",dataset)
# target = EEG_data.pop('Sensor value')

# dataset = tf.data.Dataset.from_tensor_slices((EEG_data.values, target.values))
for feat, targ in dataset.take(60):
  print ('Features: {}, Target: {}'.format(feat, targ))

# print(tf.constant(EEG_data['Sensor value']))
train_dataset = dataset.shuffle(len(EEG_data)).batch(1)

# plt.imshow(EEG_data)
# plt.show()

def get_compiled_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(580, activation='relu'),
    tf.keras.layers.Dense(240, activation='relu'),
      tf.keras.layers.Dense(120, activation='relu'),
      tf.keras.layers.Dense(2)
  ])


  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                # loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])
  return model

model = get_compiled_model()
class_names=["Non-drowsy","Drowsy"]
model.fit(train_dataset,epochs=20)


#TESTING STARTS HERE GOOD LUCK

print("********************************    TESTING STARTED ********************************")


TEST_EEG_data = pd.DataFrame({}) ## create an empty df that will hold data from each file

directory = os.path.join("C:\\","Users\VOHRA\PycharmProjects\Analyse\input\SMNI_CMI_TEST\SMNI_CMI_TEST\co2a0000365")
for root,dirs,files in os.walk(directory):
    for file in files:
       if file.endswith(".gz"):

           test_temp_df = pd.read_csv("C:\\Users\VOHRA\PycharmProjects\Analyse\input\SMNI_CMI_TEST\SMNI_CMI_TEST\co2a0000365\\"+file,skiprows = 4,nrows=256,delimiter=' ',usecols = [3])  ## read from the file to df
           test_temp_df=test_temp_df.transpose()
           # print(test_temp_df)
           if file[3]=='a':
               test_temp_df['Group']=1
           else:
               test_temp_df['Group']=0
           TEST_EEG_data = TEST_EEG_data.append(test_temp_df)

print(TEST_EEG_data)
TEST_EEG_data.to_excel(r"C:\Users\VOHRA\PycharmProjects\Analyse\input\testFile.xlsx")   #Export to external csv
test_dataset = (
    tf.data.Dataset.from_tensor_slices(
        (
            [tf.cast(TEST_EEG_data[features].values, tf.float32)],
            [tf.cast(TEST_EEG_data['Group'].values, tf.int8)]
        )
    )
)

# plt.imshow(test_dataset)
# plt.show()


# for feat in test_dataset.take(5):
#   print ('Features: {}'.format(feat))

# print[test_dataset[0]]

test_dataset_final = test_dataset.shuffle(len(TEST_EEG_data)).batch(1)
testloss,testAcc =model.evaluate(test_dataset_final)
print("test acc:",testAcc)

#prediction
# print(test_dataset)
prediction=model.predict(test_dataset_final)
print(prediction)
l=[]
for i in range(30):
    l.append(class_names[np.argmax(prediction[0][i])])
    print(class_names[np.argmax(prediction[0][i])])
    print("---------------------")

if(testAcc>.5):
    print("THE SUBJECT IS DROWSY by:%",l.count("Drowsy")/len(l)*100,"%")
# else:
    print("THE SUBJECT IS ACTIVE by:%",(1-l.count("Drowsy")/len(l))*100,"%")


