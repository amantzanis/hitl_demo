import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
import datetime

# loading path
path = os.path.dirname(__file__)

# load csv's
unlabeled = pd.read_csv(path +'/unlabeled.csv')
labeled = pd.read_csv(path +'/labeled.csv')

# split unlabeled data into to batches to feed into model after labeling
batch1 = unlabeled[:50].copy()
batch2 = unlabeled[50:].copy()

# split labeled data into to batches to feed into model
batch3 = labeled[:50].copy()
batch4 = labeled[50:].copy()

# load model
model = tf.keras.models.load_model(path + "/furn.h5")

import streamlit as st
import pandas as pd
import os
import datetime
import tensorflow as tf
import matplotlib.pyplot as plt

def add_labels(df):
    l = []
    for i in range(len(df)):
        st.write(df.iloc[i].values)
        st.write("\n")
        st.write("Please label this instance either 1 or 0")
        x = st.number_input("Label", min_value=0, max_value=1, step=1)
        l.append(x)

    df["target"] = pd.Series(l)
    return df

def load_latest(path):
    date_format = "%Y-%m-%d %H:%M:%S"
    l = []

    for i in os.listdir(path):
        if "h5" in i:
            date = i.split(".")[0].split("_")[-1]
            st.write(date)
            try:
                date_obj = datetime.datetime.strptime(date, date_format)
            except:
                continue
            l.append(date_obj)
    sorted(l)
    latest = "heart_disease_model_" + str(sorted(l)[0]) + ".h5"

    return latest

def main():
    st.title("Streamlit Heart Disease Model Retraining App")
    
    score = {}
    s = 0
    for i, j in enumerate([batch1, batch2]):
        if s == 0:  # if in the first loop
            # Load the model
            loaded_model = tf.keras.models.load_model("heart_disease_model.h5")
        else:  # load the latest model
            path = "/Users/asteriosmantzanis/SXAIPI/"
            latest = load_latest(path)
            loaded_model = tf.keras.models.load_model("heart_disease_model.h5")
        s += 1
        
        # label new data
        df = add_labels(j)
        X = df.drop("target", axis=1)
        y = df["target"]
        
        # retrain
        loaded_model.fit(X, y, epochs=10, batch_size=32)
        
        # evaluate (you should provide X_test and y_test)
        loss, acc = loaded_model.evaluate(X_test, y_test)
        score[i] = acc
        
        # update model
        fingerprint = str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        new_name = "heart_disease_model" + "_" + fingerprint + ".h5"
        loaded_model.save(new_name)

    data = [accuracy] + list(score.values())

    # Labels for the bars
    labels = ["Initial", "1st Retraining", "2nd Retraining"]

    # Bar plot
    st.bar_chart(data, labels=labels)

if __name__ == "__main__":
    main()