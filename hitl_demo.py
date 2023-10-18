import streamlit as st
import os
import pandas as pd
import numpy as np
import tensorflow as tf


# define path
path = os.path.dirname(__file__)

# load X arrays
loaded_data = np.load(path + '/data_arrays.npz')
X_train = loaded_data['X_train']
X_test = loaded_data['X_test']
X_hold = loaded_data['X_hold']

# load y arrays
target_data = np.load(path + '/target_arrays.npz')
y_train = target_data['y_train']
y_test = target_data['y_test']
y_hold = target_data['y_hold']

# load model
loaded_model = tf.keras.models.load_model(path +'/diabetes.h5')


# Create a DataFrame with your 3 features
data = {
    "Feature1": [i for i in range(20)],
    "Feature2": [i for i in range(20)],
    "Feature3": [i for i in range(20)],
    "Target": [0 for i in range(20)]  # Initialize with 0s
}
df = pd.DataFrame(data)

st.title("Current Model Metrics:")

st.title("Explaining Predictions:")

st.title("Relabel new data here:")

# Create a checkbox to show/hide the labeling instances
show_sidebar = st.checkbox("Show Labeling Instances")

if show_sidebar:
    # Use st.sidebar for the labeling instances
    with st.sidebar:
        st.write("Instructions: Please label the 'Target' column with 0 or 1.")
        for index, row in df.iterrows():
            st.write(f"Instance {index + 1}:")
            label = st.radio(f"Label Target (0 or 1) for instance {index + 1}:", [0, 1])
            if df.at[index, 'Target'] != label:
                # Highlight the updated cell with a different background color
                st.markdown(f'<style>table tr:nth-child({index + 1}) td:nth-child(5){{background-color: blue;}}</style>', unsafe_allow_html=True)
            df.at[index, 'Target'] = label

# Use st.expander to display the original DataFrame
with st.expander("Original Data", expanded=False):
    st.write("Data to Annotate:")
    st.markdown(df.to_html(escape=False), unsafe_allow_html=True)

# Use st.expander to display the updated DataFrame
with st.expander("Updated Data", expanded=False):
    st.write("Updated DataFrame:")
    st.markdown(df.to_html(escape=False), unsafe_allow_html=True)

st.title("Track Metrics:")
