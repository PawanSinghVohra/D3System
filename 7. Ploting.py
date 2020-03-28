import numpy as np
import pandas as pd
import os
import random
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
EEG_data = pd.DataFrame({}) ## create an empty df that will hold data from each file
print("**************  LOADING TRAINING DATA   ****************")
directory = os.path.join("C:\\","Users\VOHRA\PycharmProjects\Analyse\input\SMNI_CMI_TRAIN\SMNI_CMI_TRAIN\co2a0000364")
for root,dirs,files in os.walk(directory):
    for file in files:
       if file.endswith(".gz"):

           temp_df = pd.read_csv("C:\\Users\VOHRA\PycharmProjects\Analyse\input\SMNI_CMI_TRAIN\SMNI_CMI_TRAIN\co2a0000364\\"+file,skiprows = 4,nrows=255,delimiter=' ',usecols = [3])  ## read from the file to df
           temp_df=temp_df.transpose()
           print(temp_df)
           EEG_data = EEG_data.append(temp_df)

npData=EEG_data.to_numpy()
print(npData)

print("********>>>>>\n",EEG_data)
# plt.plot(npData)
plt.plot(npData[1])
plt.plot(npData[2])
plt.plot(npData[3])
plt.plot(npData[4])
plt.plot(npData[5])
plt.plot(npData[6])
plt.show()
# plt.plot(npData)
# plt.show

