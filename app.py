import streamlit as st
import pandas as pd
import os

from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import matplotlib.pyplot as plt

# Sidebar
with st.sidebar:
    st.image("pngegg.png")
    st.header("Pengaturan")
    option = st.sidebar.selectbox("Pilih opsi", ["Classification", "Regression", "Clustering"])
    choice = st.radio("Navigation", ["Upload", "Profiling", "ML", "Download"])

if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)
        
try:
    if option == "Regression":
        from pycaret.regression import *
    elif option == "Classification":
        from pycaret.classification import *
except ImportError as e:
    st.error(f"Module import error: {e}")
    st.stop()

def create_new_dataframe(df, features, target):
    selected_columns = features + [target]
    new_df = df[selected_columns].copy()
    return new_df

if option == "Regression":
    # Menampilkan konten berdasarkan opsi yang dipilih
    if choice == "Upload":
        st.title("Upload Your Data For Modelling")
        file = st.file_uploader("Upload Dataset Here", type=["csv", "xlsx", "json", 'data'])
        if file is not None:
            file_extension = file.name.split(".")[-1]
            if file_extension == "csv" or file_extension == "data":
                df = pd.read_csv(file, index_col=None)
            elif file_extension == "xlsx":
                df = pd.read_excel(file, index_col=None)
            elif file_extension == "json":
                df = pd.read_json(file)
            else:
                st.error("Unsupported file format. Please upload a CSV, Excel, or JSON file.")
                st.stop()
            df.to_csv("sourcedata.csv", index=None)
            st.dataframe(df)

    if choice == "Profiling":
        st.title("Profiling Dataset")
        profile_report = df.profile_report()
        st_profile_report(profile_report)

    if choice == "ML":
        st.title("REGRESI OKE!")
        features = st.multiselect("Select Features", df.columns)
        target = st.selectbox("Select Target", df.columns)
        new_df = create_new_dataframe(df, features, target)
        new_df
        if st.button("Train Model"):
            st.subheader("Selected Features and Target:")
            st.write("Features:", features)
            st.write("Target:", target)
            with st.spinner("Running Machine Learning Experiment..."):
                setup(new_df, target=target, verbose=False)
                setup_df = pull()
            st.info("This is ML experiment settings")
            st.dataframe(setup_df)
            with st.spinner("Comparing Models..."):
                best_model = compare_models()
                compare_df = pull()
            st.info("This is the ML models")
            st.dataframe(compare_df)
            print(best_model)
            save_model(best_model, 'regression_model')

    if choice == "Download":
        with open("regression_model.pkl", 'rb') as f:
            st.download_button("Download Model", f, "regression_model.pkl")

if option == "Classification":
    # Menampilkan konten berdasarkan opsi yang dipilih
    if choice == "Upload":
        st.title("Upload Your Data For Modelling")
        file = st.file_uploader("Upload Dataset Here", type=["csv", "xlsx", "json", "data"])
        if file is not None:
            file_extension = file.name.split(".")[-1]
            if file_extension == "csv" or file_extension == "data":
                df = pd.read_csv(file, index_col=None)
            elif file_extension == "xlsx":
                df = pd.read_excel(file, index_col=None)
            elif file_extension == "json":
                df = pd.read_json(file)
            else:
                st.error("Unsupported file format. Please upload a CSV, Excel, or JSON file.")
                st.stop()
            df.to_csv("sourcedata.csv", index=None)
            st.dataframe(df)

    if choice == "Profiling":
        st.title("Profiling Dataset")
        profile_report = df.profile_report()
        st_profile_report(profile_report)

    if choice == "ML":
        st.title("Machine Learning OKE!")
        features = st.multiselect("Select Features", df.columns)
        target = st.selectbox("Select Target", df.columns)
        new_df = create_new_dataframe(df, features, target)
        new_df
        if st.button("Train Model"):
            st.subheader("Selected Features and Target:")
            st.write("Features:", features)
            st.write("Target:", target)
            with st.spinner("Running Machine Learning Experiment..."):
                setup(new_df, target=target, verbose=False)
                setup_df = pull()
            st.info("This is ML experiment settings")
            st.dataframe(setup_df)
            with st.spinner("Comparing Models..."):
                best_model = compare_models()
                compare_df = pull()

            st.info("This is the ML models")
            st.dataframe(compare_df)
            print(best_model)

            st.subheader("Model Visualization:")

            st.write("AUC Scores:")
            try:
                plot_model(best_model, plot = 'auc', display_format='streamlit')
            except Exception as e:
                st.warning("Cannot display the plot, possibly because the model does not support it!")

            st.write("Boundary:")
            try:
                plot_model(best_model, plot = 'boundary', display_format='streamlit')
            except Exception as e:
                st.warning("Cannot display the plot, possibly because the model does not support it!")

            st.write("Confusion Matrix:")
            try:
                plot_model(best_model, plot = 'confusion_matrix', display_format='streamlit')
            except Exception as e:
                st.warning("Cannot display the plot, possibly because the model does not support it!")

            st.write("Precision-Recall Curve:")
            try:
                plot_model(best_model, plot = 'pr', display_format='streamlit')
            except Exception as e:
                st.warning("Cannot display the plot, possibly because the model does not support it!")

            st.write("Feature Importance:")
            try:
                plot_model(best_model, plot = 'feature', display_format='streamlit')
            except Exception as e:
                st.warning("Cannot display the plot, possibly because the model does not support it!")

            save_model(best_model, 'classification_model')

    if choice == "Download":
        with open("classification_model.pkl", 'rb') as f:
            st.download_button("Download Model", f, "classification_model.pkl")
