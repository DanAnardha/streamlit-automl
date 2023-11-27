import streamlit as st
import pandas as pd
import os

from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import matplotlib.pyplot as plt
from weasyprint import HTML
import webbrowser

if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)

def create_new_dataframe(df, features, target):
    selected_columns = features + [target]
    new_df = df[selected_columns].copy()
    return new_df

def profiling():
    st.title("Profiling Dataset")
    profile_report = df.profile_report()
    profile_report.to_file("report.html")
    # st_profile_report(profile_report)

st.set_page_config(page_title='The Machine Learning Algorithm Comparison App',
    layout='wide')
# Sidebar
with st.sidebar:
    st.image("pngegg.png")
    option = st.sidebar.selectbox("Options", ["Classification", "Regression", "Clustering"])
    st.header('1. Upload your CSV data')
    file = st.sidebar.file_uploader("Upload Dataset Here:", type=["csv", "xlsx", "json", 'data'])
    # st.header("Pengaturan")
    choice = st.radio("Navigation", ["Upload", "Profiling", "ML", "Download"])

try:
    if option == "Regression":
        from pycaret.regression import *
    elif option == "Classification":
        from pycaret.classification import *
except ImportError as e:
    st.error(f"Module import error: {e}")
    st.stop()

if option == "Regression":
    # Menampilkan konten berdasarkan opsi yang dipilih
    if choice == "Upload":
        st.write("""
            # The Machine Learning Algorithm Comparison App

            In this implementation, the **lazypredict** library is used for building several machine learning models at once.

            Developed by: [DanAnardha](https://www.github.com/DanAnardha)

            """)
        if file is not None:
            file_extension = file.name.split(".")[-1]
            if file_extension == "csv" or file_extension == "data":
                df = pd.read_csv(file, index_col=None)
            elif file_extension == "xlsx":
                df = pd.read_excel(file, index_col=None)
            elif file_extension == "json":
                df = pd.read_json(file)
            else:
                st.sidebar.error("Unsupported file format. Please upload a CSV, Excel, or JSON file.")
                st.sidebar.stop()
            
            df.to_csv("sourcedata.csv", index=None)
            st.dataframe(df)
        else:
            st.info('Awaiting for CSV file to be uploaded.')
            if st.button('Press to use Example Dataset'):
                if option == "Regression":
                    df = pd.read_csv('example_datasets/regression_example.csv')
                elif option == "Classification":
                    df = pd.read_csv('example_datasets/classification_example.csv')
                st.dataframe(df)
        
        path_to_html = "dd/output_profile.html" 
        with open(path_to_html,'r') as f: 
            html_data = f.read()
        if st.button("Buka HTML di Tab Baru", disabled=True):
            temp_html_path = "temp.html"
            with open(temp_html_path, 'w') as temp_file:
                temp_file.write(html_data)
            webbrowser.open_new_tab(temp_html_path)
        # if st.button("Analyze Dataset"):
        #     profiling()

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
