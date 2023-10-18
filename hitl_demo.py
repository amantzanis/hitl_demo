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

# Use st.sidebar for the labeling instances
with st.sidebar:
    st.write("Instructions: Please label the 'Target' column with 0 or 1.")
    for index, row in df.iterrows():
        st.write(f"Instance {index + 1}:")
        label = st.radio(f"Label Target (0 or 1) for instance {index + 1}:", [0, 1])
        df.at[index, 'Target'] = label

# Display the DataFrame in the main content area
st.write("Data to Annotate:")
st.dataframe(df)

# Show the updated DataFrame
st.write("Updated DataFrame:")
st.dataframe(df)
