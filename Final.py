import numpy as np
import pandas as pd
import os
import random
import tensorflow as tf

from flask import Flask,render_template,request

app=Flask(__name__)

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/')
def index():
    model = tf.keras.models.load_model("model.h5")
    features = []
    for i in range(0, 255):
        features.append(i)

    class_names=["Non-drowsy","Drowsy"]
    TEST_EEG_data = pd.DataFrame({})  ## create an empty df that will hold data from each file
    print("****************   LOADING TESTING DATA *******************")
    directory = os.path.join("C:\\",
                             "Users\VOHRA\PycharmProjects\Analyse\input\SMNI_CMI_TEST\SMNI_CMI_TEST\co2a0000372")
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".gz"):

                test_temp_df = pd.read_csv(
                    "C:\\Users\VOHRA\PycharmProjects\Analyse\input\SMNI_CMI_TEST\SMNI_CMI_TEST\co2a0000372\\" + file,
                    skiprows=4, delimiter=' ', nrows=256, usecols=[3])  ## read from the file to df
                test_temp_df = test_temp_df.transpose()
                # print(test_temp_df)
                if file[3] == 'a':
                    test_temp_df['Group'] = 1
                else:
                    test_temp_df['Group'] = 0
                TEST_EEG_data = TEST_EEG_data.append(test_temp_df)

    # TEST_EEG_data.to_excel(r"C:\Users\VOHRA\PycharmProjects\Analyse\input\testFile.xlsx")   #Export to external csv
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

    for feat in test_dataset.take(1):
        for i in feat:
            print('Features: {}'.format(i))

    # print[test_dataset[0]]
    test_dataset_final = test_dataset.shuffle(len(TEST_EEG_data)).batch(1)
    testloss, testAcc = model.evaluate(test_dataset_final)
    print("test acc:", testAcc)

    print("********************   PREDICTION *******************")
    # prediction
    # print(test_dataset)

    print(test_dataset_final)
    prediction = model.predict(test_dataset_final)
    print(prediction[0])
    l = []
    for i in range(30):
        l.append(class_names[np.argmax(prediction[0][i])])
        print(class_names[np.argmax(prediction[0][i])])
    d = l.count("Drowsy") / len(l) * 100
    n = (1 - l.count("Drowsy") / len(l)) * 100
    print("THE SUBJECT IS DROWSY by:%", l.count("Drowsy") / len(l) * 100, "%")
    print("THE SUBJECT IS ACTIVE by:%", (1 - l.count("Drowsy") / len(l)) * 100, "%")


    return render_template('index.html',d=str(d),n=str(n))
if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')