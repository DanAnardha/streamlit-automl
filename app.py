import streamlit as st
import pandas as pd
import os

from pandas_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from streamlit_pandas_profiling import st_profile_report
import matplotlib.pyplot as plt
from pathlib import Path
from io import BytesIO

st.set_page_config(page_title='The Machine Learning Algorithm Comparison App',
    layout='wide')

def df_supervised(df, features, target):
    selected_columns = features + [target]
    new_df = df[selected_columns].copy()
    return new_df

def df_unsupervised(df, features):
    selected_columns = features
    new_df = df[selected_columns].copy()
    return new_df

def df_report(df):
    st.subheader('2. Dataset Profiling (Optional)')
    if st.checkbox("Analyze Dataset (may take a while)"):
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

def reg_automl(split_size, seed_number):
    st.subheader("Selected Features and Target:")
    st.write("Features:", features)
    st.write("Target:", target)
    if "model_saved" not in st.session_state:
        with st.spinner("Running Machine Learning Experiment..."):
            setup(new_df, target=target, train_size=split_size, session_id=seed_number, verbose=False)
            setup_df = pull()
        st.info("This is ML experiment settings")
        st.dataframe(setup_df)
        with st.spinner("Comparing Models..."):
            best_model = compare_models()
            compare_df = pull()
        st.info("This is the ML models")
        st.dataframe(compare_df)
        best_model
        save_model(best_model, 'regression_model')
        st.session_state.model_saved = True
        st.subheader("4. Model Visualization")
        plot_reg(best_model)

def class_automl(split_size, seed_number):
    st.subheader("Selected Features and Target:")
    st.write("Features:", features)
    st.write("Target:", target)
    if "model_saved" not in st.session_state:
        with st.spinner("Running Machine Learning Experiment..."):
            setup(new_df, target=target, train_size=split_size, session_id=seed_number, verbose=False)
            setup_df = pull()
        st.info("This is ML experiment settings")
        st.dataframe(setup_df)
        with st.spinner("Comparing Models..."):
            best_model = compare_models()
            compare_df = pull()
        st.info("This is the ML models")
        st.dataframe(compare_df)
        best_model
        save_model(best_model, 'classification_model')
        st.session_state.model_saved = True
        st.subheader("4. Model Visualization")
        plot_class(best_model)

def cluster_automl(n_cluster):
    st.subheader("Selected Features")
    st.write("Features:", features)
    if "model_saved" not in st.session_state:
        with st.spinner("Running Machine Learning Experiment..."):
            setup(new_df, verbose=False)
            setup_df = pull()
        st.info("This is ML experiment settings")
        st.dataframe(setup_df)
        with st.spinner("Creating Models..."):
            model = create_model('kmeans', num_clusters=n_cluster)
        st.info("This is the ML model")
        model
        save_model(model, 'clustering_model')
        st.session_state.model_saved = True
        st.subheader("4. Model Visualization")
        plot_cluster(model)

def upload_test():
    st.subheader('1. Upload Predict Dataset')
    widget_id = (id for id in range(1, 100_00))
    st.info('Awaiting for testing dataset file to be uploaded (dataset you want to predict, must contain all features columns in model).')
    file_test = st.file_uploader("Upload Testing Dataset Here:", key=next(widget_id), type=["csv", "xlsx", "json", 'data'])
    if st.checkbox('Press to use Example Dataset'):
        if option == "Regression":
            st.markdown('The [University Admissions](https://www.kaggle.com/code/yogesh239/analysis-of-university-admissions-data) dataset is used as the example.')
            df_test = pd.read_csv('example_datasets/regression_example.csv')
        elif option == "Classification":
            st.markdown('The [Diabetes](https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset) dataset is used as the example.')
            df_test = pd.read_csv('example_datasets/classification_example.csv')
        elif option == "Clustering":
            st.markdown('The [Facebook Live Sellers in Thailand](http://archive.ics.uci.edu/dataset/488/facebook+live+sellers+in+thailand) dataset is used as the example.')
            df_test = pd.read_csv('example_datasets/clustering_example.csv')
        df_test.to_csv("sourcedata_test.csv", index=None)
        st.dataframe(df_test)
        st.info(f"Dataset dimension: {df_test.shape}")
        return df_test

    else:
        if file_test is not None:
            file_extension = file_test.name.split(".")[-1]
            if file_extension == "csv" or file_extension == "data":
                df_test = pd.read_csv(file_test, index_col=None)
            elif file_extension == "xlsx":
                df_test = pd.read_excel(file_test, index_col=None)
            elif file_extension == "json":
                df_test = pd.read_json(file_test)
            else:
                st.error("Unsupported file format. Please upload a CSV, Excel, or JSON file.")
                st.stop()
            st.markdown('The uploaded file is used as the dataset.')
            df_test.to_csv("sourcedata_test.csv", index=None)
            st.dataframe(df_test)
            st.info(f"Dataset dimension: {df_test.shape}")
            return df_test


def predict(model, df_test):
    st.write('This is your predicted data')
    pred = predict_model(model, df_test)
    st.dataframe(pred)
    st.subheader('4. Download Data')
    file_format = st.selectbox("Select format:", ['CSV', "Excel"])
    if file_format == "CSV":
        pred.to_csv('download_predict/predict.csv', index = False)
        st.download_button(label="Download", data=pred.to_csv(index=False).encode('utf-8'), file_name='predict.csv', key='download_button_csv')
    else:
        st.download_button(label="Download", data=pred.to_excel(index=False).encode('utf-8'), file_name='predict.xlsx', key='download_button_excel')

def plot_class(best_model):
    st.write("AUC Scores:")
    st.info('AUC represents the degree or measure of separability.')
    try:
        plot_model(best_model, plot = 'auc', display_format='streamlit')
    except Exception as e:
        st.warning("Cannot display the plot, possibly because the model does not support it.")

    st.write("Boundary:")
    st.info('Separates classes in machine learning visualizations.')
    try:
        plot_model(best_model, plot = 'boundary', display_format='streamlit')
    except Exception as e:
        st.warning("Cannot display the plot, possibly because the model does not support it.")

    st.write("Confusion Matrix:")
    st.info('Evaluates model performance by comparing predicted and actual class labels visually.')
    try:
        plot_model(best_model, plot = 'confusion_matrix', display_format='streamlit')
    except Exception as e:
        st.warning("Cannot display the plot, possibly because the model does not support it.")

    st.write("Precision-Recall Curve:")
    st.info('Shows trade-off between precision and recall for classification models.')
    try:
        plot_model(best_model, plot = 'pr', display_format='streamlit')
    except Exception as e:
        st.warning("Cannot display the plot, possibly because the model does not support it.")

    st.write("Decision Tree:")
    st.info('Shows tree-like model structure for predicting numerical outcomes.')
    try:
        plot_model(best_model, plot = 'tree', display_format='streamlit')
    except Exception as e:
        st.warning("Cannot display the plot, possibly because the model does not support it.")

def plot_reg(best_model):
    st.write("Learning Curve:")
    st.info('Shows representation of model performance vs. training data size.')
    try:
        plot_model(best_model, plot = 'learning', display_format='streamlit')
    except Exception as e:
        st.warning("Cannot display the plot, possibly because the model does not support it.")

    st.write("Residual:")
    st.info('Shows the difference between actual and predicted values in regression analysis.')
    try:
        plot_model(best_model, display_format='streamlit')
    except Exception as e:
        st.warning("Cannot display the plot, possibly because the model does not support it.")

    st.write("Prediction Error:")
    st.info('Shows the distribution of residuals for model evaluation.')
    try:
        plot_model(best_model, plot = 'error', display_format='streamlit')
    except Exception as e:
        st.warning("Cannot display the plot, possibly because the model does not support it.")

def plot_cluster(model):
    st.write("Elbow:")
    st.info('Helps find optimal number of clusters by identifying the "elbow" point visually.')
    try:
        plot_model(model, plot = 'elbow', display_format='streamlit')
    except Exception as e:
        st.warning("Cannot display the plot, possibly because the model does not support it.")

    st.write("Cluster:")
    st.info('Visual representation of grouped data points, showing distinct clusters based on similarities or patterns.')
    try:
        plot_model(model, plot = 'cluster', display_format='streamlit')
    except Exception as e:
        st.warning("Cannot display the plot, possibly because the model does not support it.")

    st.write("Distribution:")
    st.info('Displays the distribution of a variable, highlighting its central tendency and spread.')
    try:
        plot_model(model, plot = 'distribution', display_format='streamlit')
    except Exception as e:
        st.warning("Cannot display the plot, possibly because the model does not support it.")

    st.write("Silhouette:")
    st.info('Measures cluster cohesion and separation, helping assess clustering quality and structure.')
    try:
        plot_model(model, plot = 'silhouette', display_format='streamlit')
    except Exception as e:
        st.warning("Cannot display the plot, possibly because the model does not support it.")

    st.write("Distance:")
    st.info('Illustrates distances between data points, aiding analysis of cluster cohesion and separation.')
    try:
        plot_model(model, plot = 'distance', display_format='streamlit')
    except Exception as e:
        st.warning("Cannot display the plot, possibly because the model does not support it.")

# Sidebar
with st.sidebar:
    st.image("pngegg.png")
    action = st.sidebar.selectbox('Action', ['Create Model', 'Test Model'])
    option = st.sidebar.selectbox("Options", ["Classification", "Regression", "Clustering"])
    if action == 'Create Model' and (option == 'Regression' or option == 'Classification'):
        st.header('Upload your dataset')
        file = st.sidebar.file_uploader("Upload Dataset Here:", type=["csv", "xlsx", "json", 'data'], key='file_upload1')
        st.header('Set Parameters')
        split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5) / 100
        seed_number = st.sidebar.slider('Set the random seed number', 1, 100, 42, 1)
    elif action == 'Create Model' and option == 'Clustering':
        st.header('Upload your dataset')
        file = st.sidebar.file_uploader("Upload Dataset Here:", type=["csv", "xlsx", "json", 'data'], key='file_upload2')
    elif action == 'Test Model':
        st.header('Upload your model')
        file = st.sidebar.file_uploader("Upload Model Here:", type=["pkl"], key='file_upload_model')

try:
    if option == "Regression":
        from pycaret.regression import *
    elif option == "Classification":
        from pycaret.classification import *
    elif option == "Clustering":
        from pycaret.clustering import *
except ImportError as e:
    st.error(f"Module import error: {e}")
    st.stop()

if action == 'Create Model':
    st.write("""
        # The Auto Machine Learning App

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
            st.subheader("3. Train Regression Model")
            st.info('Choose features and targets carefully and systematically.')
            features = st.multiselect("Select Features", df.columns)
            target = st.selectbox("Select Target", df.columns)
            try:
                new_df = df_supervised(df, features, target)
                new_df
            except Exception as e:
                st.warning("Cannot display the dataframe, possibly because you selected the same feature and target.")
            st.info('You can set the training data split ratio and random seed number on the sidebar.')
            if st.button("Train Model"):
                reg_automl(split_size, seed_number)
                st.subheader('5. Download Model')
                st.info('The model you downloaded is the model with the lowest error.')
                st.warning('Refresh this site when done training. (This site will refreshed when you click Download Model)')
                with open("regression_model.pkl", 'rb') as f:
                    st.download_button("Download Model", f, "regression_model.pkl")
        elif option == "Classification":
            st.subheader("3. Train Classification Model")
            st.info('Choose features and targets carefully and systematically.')
            features = st.multiselect("Select Features", df.columns)
            target = st.selectbox("Select Target", df.columns)
            try:
                new_df = df_supervised(df, features, target)
                new_df
            except Exception as e:
                st.warning("Cannot display the dataframe, possibly because you selected the same feature and target.")
            st.info('You can set the training data split ratio and random seed number on the sidebar.')
            if st.button("Train Model"):
                class_automl(split_size, seed_number)
                st.subheader('5. Download Model')
                st.info('The model you downloaded is the model with the highest accuracy.')
                st.warning('Refresh this site when done training. (This site will refreshed when you click Download Model)')
                with open("classification_model.pkl", 'rb') as f:
                    st.download_button("Download Model", f, "classification_model.pkl")
        elif option == 'Clustering':
            st.subheader("3. Train Clustering Model")
            st.info('Choose features and carefully and systematically.')
            n_cluster = st.slider('Choose the number of clusters', 2, 20, 3, 1)
            features = st.multiselect("Select Features", df.columns)
            try:
                new_df = df_unsupervised(df, features)
                new_df
            except Exception as e:
                st.warning("Cannot display the dataframe, possibly because you selected the same feature and target.")
            if st.button("Train Model"):
                cluster_automl(n_cluster)
                st.subheader('5. Download Model')
                st.warning('Refresh this site when done training. (This site will refreshed when you click Download Model)')
                with open("clustering_model.pkl", 'rb') as f:
                    st.download_button("Download Model", f, "clustering_model.pkl")
    else:
        st.info('Awaiting for dataset file to be uploaded.')
        if st.checkbox('Press to use Example Dataset'):
            if option == "Regression":
                st.markdown('The [University Admissions](https://www.kaggle.com/code/yogesh239/analysis-of-university-admissions-data) dataset is used as the example.')
                df = pd.read_csv('example_datasets/regression_example.csv')
            elif option == "Classification":
                st.markdown('The [Diabetes](https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset) dataset is used as the example.')
                df = pd.read_csv('example_datasets/classification_example.csv')
            elif option == "Clustering":
                st.markdown('The [Facebook Live Sellers in Thailand](http://archive.ics.uci.edu/dataset/488/facebook+live+sellers+in+thailand) dataset is used as the example.')
                df = pd.read_csv('example_datasets/clustering_example.csv')
            df.to_csv("sourcedata.csv", index=None)
            st.dataframe(df)
            st.info(f"Dataset dimension: {df.shape}")
            df_report(df)
            if option == "Regression":
                st.subheader("3. Train Regression Model")
                st.info('Choose features and targets carefully and systematically.')
                features = st.multiselect("Select Features", df.columns)
                target = st.selectbox("Select Target", df.columns)
                try:
                    new_df = df_supervised(df, features, target)
                    new_df
                except Exception as e:
                    st.warning("Cannot display the dataframe, possibly because you selected the same feature and target.")
                st.info('You can set the training data split ratio and random seed number on the sidebar.')
                if st.button("Train Model"):
                    reg_automl(split_size, seed_number)
                    st.subheader('5. Download Model')
                    st.info('The model you downloaded is the model with the lowest error.')
                    st.warning('Refresh this site when done training. (This site will refreshed when you click Download Model)')
                    with open("regression_model.pkl", 'rb') as f:
                        st.download_button("Download Model", f, "regression_model.pkl")
            elif option == "Classification":
                st.subheader("3. Train Classification Model")
                st.info('Choose features and targets carefully and systematically.')
                features = st.multiselect("Select Features", df.columns)
                target = st.selectbox("Select Target", df.columns)
                try:
                    new_df = df_supervised(df, features, target)
                    new_df
                except Exception as e:
                    st.warning("Cannot display the dataframe, possibly because you selected the same feature and target.")
                st.info('You can set the training data split ratio and random seed number on the sidebar.')
                if st.button("Train Model"):
                    class_automl(split_size, seed_number)
                    st.subheader('5. Download Model')
                    st.info('The model you downloaded is the model with the highest accuracy.')
                    st.warning('Refresh this site when done training. (This site will refreshed when you click Download Model)')
                    with open("classification_model.pkl", 'rb') as f:
                        st.download_button("Download Model", f, "classification_model.pkl")
            elif option == 'Clustering':
                st.subheader("3. Train Clustering Model")
                st.info('Choose features and carefully and systematically.')
                n_cluster = st.slider('Choose the number of clusters', 2, 20, 3, 1)
                features = st.multiselect("Select Features", df.columns)
                try:
                    new_df = df_unsupervised(df, features)
                    new_df
                except Exception as e:
                    st.warning("Cannot display the dataframe, possibly because you selected the same feature and target.")
                if st.button("Train Model"):
                    cluster_automl(n_cluster)
                    st.subheader('5. Download Model')
                    st.warning('Refresh this site when done training. (This site will refreshed when you click Download Model)')
                    with open("clustering_model.pkl", 'rb') as f:
                        st.download_button("Download Model", f, "clustering_model.pkl")
else:
    st.write("""
        # The Auto Machine Learning App

        In this implementation, the **PyCaret** library is used for building several machine learning models at once.

        Developed by: [DanAnardha](https://www.github.com/DanAnardha)

        """)
    if file is not None:
        file_extension = file.name.split(".")[-1]
        file_path = Path(file.name)
        file_name = file_path.stem
        file_extension = file_path.suffix
        with open(file_name + file_extension, "wb") as f:
            f.write(file.read())
        try:
            if option == "Regression":
                model = load_model(file_name)
            elif option == "Classification":
                model = load_model(file_name)
            elif option == "Clustering":
                model = load_model(file_name)
        except Exception as e:
            st.error(f"Cannot load the model, Error {e}")
        df_test = upload_test()
        st.subheader('2. Predict Model')
        st.info('Uploaded dataset must include all features from trained model')
        if st.button('Predict'):
            predict(model, df_test)
    else:
        if st.sidebar.checkbox('Press to use Example Model'):
            if option == "Regression":
                model = load_model('example_models/regression_model')
            elif option == "Classification":
                model = load_model('example_models/classification_model')
            elif option == "Clustering":
                model = load_model('example_models/clustering_model')
            df_test = upload_test()
            st.subheader('2. Predict Model')
            st.info('Uploaded dataset must include all features from trained model')
            if st.button('Predict'):
                predict(model, df_test)
