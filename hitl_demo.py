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

# Display the DataFrame in Streamlit
st.write(df)

st.write("Instructions: Please label the 'Target' column with 0 or 1.")

# Create a form to update the target labels
for index, row in df.iterrows():
    st.write(f"Instance {index + 1}:")
    label = st.radio(f"Label Target (0 or 1) for instance {index + 1}:", [0, 1])
    df.at[index, 'Target'] = label

# Show the updated DataFrame
st.write("Updated DataFrame:")
st.write(df)
