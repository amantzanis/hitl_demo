import streamlit as st
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# ignore streamlit warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

# init global variable model_retrained
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

# load model once for predictions without human feedback
loaded_model = tf.keras.models.load_model(path +'/diabetes.h5')

# load model once more for predictions with human feedback
loaded_model1 = tf.keras.models.load_model(path +'/diabetes.h5')

# load holdout data
df = pd.read_csv(path + '/holdout.csv')

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
                st.markdown(f'<style>table tr:nth-child({index + 1}) td:nth-child(5){{background-color: Plum;}}</style>', unsafe_allow_html=True)
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
            X = df.drop('Target', axis=1)
            y = df['Target']
            X1 = updated_df.drop('Target', axis=1)
            y1 = updated_df['Target']
            # Train the model
            # Define early stopping
            es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            with st.spinner(text='In progress'):
                loaded_model.fit(X, y, epochs=100, batch_size=10, validation_split=0.1, callbacks=[es])
                loaded_model1.fit(X1, y1, epochs=100, batch_size=10, validation_split=0.1, callbacks=[es])
            model_retrained = True
            
        if model_retrained:
            st.success("Model has been retrained!")
            st.balloons()
    
    # Define a function to plot the bar chart
    def plot_accuracy_bar(initial_accuracy, updated_accuracy_w, updated_accuracy_w_hf):
    
        fig, ax = plt.subplots()
        metrics = ['Initial Accuracy','without Human Feedback', 'with Human Feedback']
        accuracy_values = [initial_accuracy, accuracy_wo_hf, accuracy_new]
        ax.bar(metrics, accuracy_values)
        ax.set_ylabel('Accuracy')
        ax.tick_params(axis='x', rotation=45, labelsize = 9)
        
        # Annotate the bars with percentages
        for i, v in enumerate(accuracy_values):
            ax.text(i, v - 0.1, f'{v:.2%}', ha='center', va='bottom', fontsize=12, color='white')
        st.pyplot()
    
    st.title("Track Metrics:")
    with st.expander("Updated Metrics:", expanded=False):
        # Calculate the updated accuracy after retraining the model
        if df.equals(updated_df):
            _, accuracy_wo_hf = loaded_model.evaluate(X_test, y_test)
            accuracy_new = accuracy_wo_hf
        else:
            _, accuracy_wo_hf = loaded_model.evaluate(X_test, y_test)
            _, accuracy_new = loaded_model1.evaluate(X_test, y_test)
        plot_accuracy_bar(accuracy, accuracy_wo_hf, accuracy_new)

    st.title("Update Model:")
    with st.expander("Choose a model:", expanded=False):
        # Create buttons to choose which model to keep
        if model_retrained:
            model_choice = st.radio("Choose a Model to Keep:", [None, "Original Model", "Retrained Model", "Retrained Model with Human Feedback"])
        
            if model_choice == "Original Model":
                st.write("You have chosen to keep the original model.")
                # Add code here to save the original model if needed.
            elif model_choice == "Retrained Model":
                st.write("You have chosen to keep the retrained model without human feedback.")
                # Add code here to save the retrained model without human feedback if needed.
            elif model_choice == "Retrained Model with Human Feedback":
                st.write("You have chosen to keep the retrained model with human feedback.")
                # Add code here to save the retrained model with human feedback if needed.
