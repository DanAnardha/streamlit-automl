import streamlit as st
import pandas as pd
import os

from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from pycaret.regression import *

# Sidebar
with st.sidebar:
    st.image("pngegg.png")
    st.header("Pengaturan")
    option = st.sidebar.selectbox("Pilih opsi", ["Classification", "Regression", "Clustering"])
    choice = st.radio("Navigation", ["Upload", "Profiling", "ML", "Download"])

if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)
        
if option == "Regression":
    # Menampilkan konten berdasarkan opsi yang dipilih
    if choice == "Upload":
        st.title("Upload Your Data For Modelling")
        file = st.file_uploader("Upload Dataset Here")
        if file:
            df = pd.read_csv(file, index_col=None)
            df.to_csv("sourcedata.csv", index=None)
            st.dataframe(df)

    if choice == "Profiling":
        st.title("Profiling Dataset")
        profile_report = df.profile_report()
        st_profile_report(profile_report)

    if choice == "ML":
        st.title("Machine Learning OKE!")
        target = st.selectbox("Select your target", df.columns)
        if st.button("Train Model"):
            with st.spinner("Running Machine Learning Experiment..."):
                setup(df, target=target, verbose=False)
                setup_df = pull()
            st.info("This is ML experiment settings")
            st.dataframe(setup_df)
            with st.spinner("Comparing Models..."):
                best_model = compare_models()
                compare_df = pull()

            st.info("This is the ML models")
            st.dataframe(compare_df)
            best_model
            save_model(best_model, 'model')

    if choice == "Download":
        with open("model.pkl", 'rb') as f:
            st.download_button("Download Model", f, "trained_model.pkl")