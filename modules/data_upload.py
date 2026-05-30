import streamlit as st
import pandas as pd

def upload_data():
    st.subheader("📂 Upload Dataset")

    file = st.file_uploader(
        "Upload your CSV file",
        type=["csv"]
    )

    if file is not None:
        df = pd.read_csv(file)

        st.success("Dataset loaded successfully!")
        st.write("Preview of dataset:")
        st.dataframe(df.head())

        return df

    return None
