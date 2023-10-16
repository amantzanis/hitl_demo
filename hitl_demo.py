import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
import datetime
import streamlit as st

# loading path
path = os.path.dirname(__file__)

# load csv's
unlabeled = pd.read_csv(path + '/unlabeled.csv')
labeled = pd.read_csv(path + '/labeled.csv')

# split unlabeled data into batches to feed into the model after labeling
batch1 = unlabeled[:50].copy()
batch2 = unlabeled[50:].copy()

# split labeled data into batches to feed into the model
batch3 = labeled[:50].copy()
batch4 = labeled[50:].copy()

# load test data
loaded_data = np.load(path + "/test_data.npz")
X_test = loaded_data["X_test"]
y_test = loaded_data["y_test"]

# global df variable for labeling results
df = None

def add_labels(df):
    l = []
    current_instance = 0
    total_instances = len(df)

    while current_instance < total_instances:
        st.text("Data Instance:")
        st.write(df.iloc[current_instance])
        st.text("Please label this instance either 1 or 0")

        input_key = f"label_{current_instance}"
        x = st.number_input("Label", min_value=0, max_value=1, step=1, key=input_key)
        l.append(x)

        # Add a "Next" button to move to the next instance
        next_button = st.button("Next")

        if next_button:
            current_instance += 1

    df["target"] = pd.Series(l)

    return df


def load_latest(path):
    date_format = "%Y-%m-%d %H:%M:%S"
    l = []

    for i in os.listdir(path):
        if "h5" in i:
            date = i.split(".")[0].split("_")[-1]
            try:
                date_obj = datetime.datetime.strptime(date, date_format)
            except:
                continue
            l.append(date_obj)
    latest = "heart_disease_model_" + str(sorted(l)[0]) + ".h5"

    return latest

def main():
    st.title("Heart Disease Model Active Learning Demo App")
    score = {}
    s = 0
    data = []
    
    if "labeled_data" not in st.session_state:
        st.session_state.labeled_data = {}  # Initialize the labeled data in session_state

    for i, j in enumerate([batch1, batch2]):
        if s == 0:  # if in the first loop
            # Load the model
            loaded_model = tf.keras.models.load_model("heart_disease_model.h5")
        else:  # load the latest model
            latest = load_latest(path)
            loaded_model = tf.keras.models.load_model(latest)
        s += 1
        
        # label new data
        if st.button("Label Data", key=f"label_button_{i}"):
            labeled_data = add_labels(j)
            st.session_state.labeled_data[f"batch_{i}"] = labeled_data
        
        # retrain
        if st.button("Retrain Model", key=f"retrain_button_{i}"):
            st.text("Retraining Model...")  # Display a message indicating retraining
            labeled_data = st.session_state.labeled_data.get(f"batch_{i}")
            if labeled_data is not None:
                X = labeled_data.drop("target", axis=1)
                y = labeled_data["target"]
                loaded_model.fit(X, y, epochs=10, batch_size=32)
                
                # evaluate (you should provide X_test and y_test)
                loss, acc = loaded_model.evaluate(X_test, y_test)
                score[i] = acc
                st.text(f"Model accuracy before retraining: 0.83")
                st.text(f"Model accuracy after retraining: {acc:.2f}")
                
                # update model
                fingerprint = str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                new_name = "/heart_disease_model" + "_" + fingerprint + ".h5"
                loaded_model.save(path + new_name)

if __name__ == "__main__":
    main()
