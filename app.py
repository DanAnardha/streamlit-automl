import streamlit as st
import pandas as pd
import os

from pandas_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from streamlit_pandas_profiling import st_profile_report
import matplotlib.pyplot as plt
import webbrowser
import random


st.set_page_config(page_title='The Machine Learning Algorithm Comparison App',
    layout='wide')

def create_new_dataframe(df, features, target):
    selected_columns = features + [target]
    new_df = df[selected_columns].copy()
    return new_df

def df_report(df):
    st.subheader('2. Dataset Profiling (Optional)')
    if st.checkbox("Analyze Dataset"):
        if "profile_report" not in st.session_state:
            with st.spinner("Creating dataset profile report..."):
                profile_report = df.profile_report()
                st.session_state.profile_report = profile_report

        path_to_html = "report.html"
        with open(path_to_html, 'r') as f:
            html_data = st.session_state.profile_report.to_html()
            st.components.v1.html(html_data, height=768, scrolling=True)

        st.download_button(
            label="Download Report",
            data=html_data.encode('utf-8'),
            file_name="report.html",
            mime="text/html",
            key="download_button_key"
        )

def reg_automl():
    st.subheader("3. Training ML Model")
    features = st.multiselect("Select Features", df.columns)
    target = st.selectbox("Select Target", df.columns)
    try:
        new_df = create_new_dataframe(df, features, target)
        new_df
    except Exception as e:
        st.warning("Cannot display the dataframe, possibly because you selected the same feature and target.")

   
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

def class_automl():
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
        save_model(best_model, 'classification_model')
        plot_class(best_model)

def plot_class(best_model):

    st.subheader("Model Visualization:")

    st.write("AUC Scores:")
    try:
        plot_model(best_model, plot = 'auc', display_format='streamlit')
    except Exception as e:
        st.warning("Cannot display the plot, possibly because the model does not support it.")

    st.write("Boundary:")
    try:
        plot_model(best_model, plot = 'boundary', display_format='streamlit')
    except Exception as e:
        st.warning("Cannot display the plot, possibly because the model does not support it.")

    st.write("Confusion Matrix:")
    try:
        plot_model(best_model, plot = 'confusion_matrix', display_format='streamlit')
    except Exception as e:
        st.warning("Cannot display the plot, possibly because the model does not support it.")

    st.write("Precision-Recall Curve:")
    try:
        plot_model(best_model, plot = 'pr', display_format='streamlit')
    except Exception as e:
        st.warning("Cannot display the plot, possibly because the model does not support it.")

    st.write("Feature Importance:")
    try:
        plot_model(best_model, plot = 'feature', display_format='streamlit')
    except Exception as e:
        st.warning("Cannot display the plot, possibly because the model does not support it.")

# Sidebar
with st.sidebar:
    st.image("pngegg.png")
    option = st.sidebar.selectbox("Options", ["Classification", "Regression", "Clustering"])
    st.header('1. Upload your CSV data')
    file = st.sidebar.file_uploader("Upload Dataset Here:", type=["csv", "xlsx", "json", 'data'])
    st.header('2. Set Parameters')
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)
    seed_number = st.sidebar.slider('Set the random seed number', 1, 100, 42, 1)
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

st.write("""
    # The Machine Learning Algorithm Comparison App

    In this implementation, the **PyCaret** library is used for building several machine learning models at once.

    Developed by: [DanAnardha](https://www.github.com/DanAnardha)

    """)
st.subheader('1. Dataset')
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
    st.markdown('The uploaded file is used as the dataset.')
    df.to_csv("sourcedata.csv", index=None)
    st.dataframe(df)
    st.info(f"Dataset dimension: {df.shape}")
    df_report(df)
    if option == "Regression":
        reg_automl()
    elif option == "Classification":
        class_automl()
else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.checkbox('Press to use Example Dataset'):
        if option == "Regression":
            st.markdown('The [University Admissions](https://www.kaggle.com/code/yogesh239/analysis-of-university-admissions-data) dataset is used as the example.')
            df = pd.read_csv('example_datasets/regression_example.csv')
        elif option == "Classification":
            st.markdown('The [Diabetes](https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset) dataset is used as the example.')
            df = pd.read_csv('example_datasets/classification_example.csv')
        df.to_csv("sourcedata.csv", index=None)
        st.dataframe(df)
        st.info(f"Dataset dimension: {df.shape}")
        df_report(df)
        if option == "Regression":
            reg_automl()
        elif option == "Classification":
            class_automl()

    if choice == "Download":
        with open("regression_model.pkl", 'rb') as f:
            st.download_button("Download Model", f, "regression_model.pkl")

    if choice == "Download":
        with open("classification_model.pkl", 'rb') as f:
            st.download_button("Download Model", f, "classification_model.pkl")
