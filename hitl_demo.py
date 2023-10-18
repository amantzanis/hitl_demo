import streamlit as st
import pandas as pd

# Create a DataFrame with your 3 features
data = {
    "Feature1": [i for i in range(20)],
    "Feature2": [i for i in range(20)],
    "Feature3": [i for i in range(20)],
    "Target": [0 for i in range(20)]  # Initialize with 0s
}

# Set up the Streamlit layout
st.title("Data Annotation App")

# Create a container to hold the DataFrame
container = st.container()

# Use st.sidebar() for the selection on the left side
with st.sidebar():
    st.write("Instructions: Please label the 'Target' column with 0 or 1.")
    for index, row in df.iterrows():
        st.write(f"Instance {index + 1}:")
        label = st.radio(f"Label Target (0 or 1) for instance {index + 1}:", [0, 1])
        df.at[index, 'Target'] = label

# Use CSS to position the DataFrame on the right and make it sticky
st.markdown(
    """
    <style>
    .main {
        display: flex;
    }

    .block-container {
        flex: 1;
    }

    .block-container > div {
        position: sticky;
        top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Display the DataFrame within the container
with container:
    st.markdown("Data to Annotate:")
    st.dataframe(df)
