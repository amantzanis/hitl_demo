import streamlit as st
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import shap
import tensorflow as tf

# ignore streamlit warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

model_retrained = False

# define path
path = os.path.dirname(__file__)

# Upload the background image
image_path = path + '/sx.png'
background_image = st.image(image_path, use_column_width=False)

# Add custom CSS to set the background image
st.write(
    f"""
    <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{background_image.data}");
            background-size: auto;
            background-position: top center;
            background-repeat: no-repeat;
            width: auto;
            height: auto;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# load X arrays
loaded_data = np.load(path + '/data_arrays.npz')
X_test = loaded_data['X_test']
X_hold = loaded_data['X_hold']

# load y arrays
target_data = np.load(path + '/target_arrays.npz')
y_test = target_data['y_test']
y_hold = target_data['y_hold']

# load model
loaded_model = tf.keras.models.load_model(path +'/diabetes.h5')

# load holdout data
df = pd.read_csv(path + '/holdout.csv')

# explainer = shap.KernelExplainer(loaded_model.predict,X_train)
# shap_values = explainer.shap_values(X_test)

st.title("Human In The Loop Concept Demo")
st.write('Simple app to showcase our hitl ideas built with an open source diabetes dataset.')

st.title("Current Model Metrics:")
with st.expander("Initial Metrics:", expanded=False):
           # Evaluate the model on the test data
           loss, accuracy = loaded_model.evaluate(X_test, y_test)
           st.write(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

st.title("Explaining Predictions:")
with st.expander("Explain", expanded=False):
    # Load and display a PNG image
    image_path = path + '/mygraph.jpg'
    image = st.image(image_path, use_column_width=True)

st.title("Relabel and Retrain:")

# Create a checkbox to show/hide the labeling instances
show_sidebar = st.checkbox("Show Labeling Instances")

# if show_sidebar:
#     # Use st.sidebar for the labeling instances
#     with st.sidebar:
#         st.write("Instructions: Please label the 'Target' column with 0 or 1.")
#         target_values = df["Target"].tolist()
#         for index, row in df.iterrows():
#             st.write(f"Instance {index}:")
#             label = st.radio(f"Label Target (0 or 1) for instance {index}:", [0, 1], index=target_values[index])
#             if df.at[index, 'Target'] != label:
#                 # Highlight the updated cell with a different background color
#                 st.markdown(f'<style>table tr:nth-child({index + 1}) td:nth-child(5){{background-color: blue;}}</style>', unsafe_allow_html=True)
#             df.at[index, 'Target'] = label

# # Use st.expander to display the original DataFrame
# with st.expander("Original Data", expanded=False):
#     st.write("Data to Annotate:")
#     st.markdown(df.to_html(escape=False), unsafe_allow_html=True)

# # Use st.expander to display the updated DataFrame
# with st.expander("Updated Data", expanded=False):
#     st.write("Updated DataFrame:")
#     st.markdown(df.to_html(escape=False), unsafe_allow_html=True)

if show_sidebar:
    updated_df = df.copy()  # Create a copy of the original DataFrame to hold the updated data

    # Use st.sidebar for the labeling instances
    with st.sidebar:
        st.write("Instructions: Please label the 'Target' column with 0 or 1.")
        target_values = df["Target"].tolist()
        for index, row in df.iterrows():
            st.write(f"Instance {index}:")
            label = st.radio(f"Label Target (0 or 1) for instance {index}:", [0, 1], index=target_values[index])
            if df.at[index, 'Target'] != label:
                # Highlight the updated cell with a different background color
                st.markdown(f'<style>table tr:nth-child({index + 1}) td:nth-child(5){{background-color: blue;}}</style>', unsafe_allow_html=True)
            updated_df.at[index, 'Target'] = label

    # Use st.expander to display the original DataFrame
    with st.expander("Original Data", expanded=False):
        st.write("Data to Annotate:")
        st.markdown(df.to_html(escape=False), unsafe_allow_html=True)
    
    # Use st.expander to display the updated DataFrame
    with st.expander("Updated Data", expanded=False):
        st.write("Updated DataFrame:")
        st.markdown(updated_df.to_html(escape=False), unsafe_allow_html=True)

    with st.expander("Retrain Model", expanded=False):
        retrain_button = st.button("Retrain Model")
        if retrain_button:
            X = updated_df.drop('Target', axis=1)
            y = updated_df['Target']
            # Train the model
            # Define early stopping
            es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            loaded_model.fit(X, y, epochs=100, batch_size=10, validation_split=0.1, callbacks=[es])
            model_retrained = True
            
        if model_retrained:
            st.success("Model has been retrained!")
            st.balloons()
           
st.title("Track Metrics:")
with st.expander("Initial Metrics:", expanded=False):
           # Evaluate the new_model on the test data
           loss_new, accuracy_new = loaded_model.evaluate(X_test, y_test)
           st.write(f"Test Loss: {loss_new:.4f}, Test Accuracy: {accuracy_new:.4f}")
