import streamlit as st
import pandas as pd

# Create a DataFrame with your 3 features
data = {
    "Feature1": [i for i in range(20)],
    "Feature2": [i for i in range(20)],
    "Feature3": [i for i in range(20)],
    "Target": [0 for i in range(20)]  # Initialize with 0s
}
df = pd.DataFrame(data)

st.title("Data Annotation App")

# Create a container to hold the selection section
st.write("Instructions: Please label the 'Target' column with 0 or 1.")
for index, row in df.iterrows():
    st.write(f"Instance {index + 1}:")
    label = st.radio(f"Label Target (0 or 1) for instance {index + 1}:", [0, 1])
    df.at[index, 'Target'] = label

# Use CSS to make the DataFrame sticky on the right
st.markdown(
    """
    <style>
    .css-145kmo2 {
        position: -webkit-sticky;
        position: sticky;
        top: 50px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Display the DataFrame
st.write("Data to Annotate:")
st.dataframe(df, height=600)
