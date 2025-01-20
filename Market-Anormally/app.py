import streamlit as st
import pandas as pd
import pickle
import os
import utils as ut

# Helper function to load .pkl files
def load_pickle(file_path):
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        st.error(f"Error loading the file: {e}")
        return None

# Streamlit App
st.title("Dynamic Stock Market Predictor 🧠📈")
st.sidebar.title("Configuration")

# Example Folders
csv_folder = "stock-data"
pkl_folder = "ML-files"

# Load example files from folders
example_csv_files = [f for f in os.listdir(csv_folder) if f.endswith(".csv")]
example_pkl_files = [f for f in os.listdir(pkl_folder) if f.endswith(".pkl")]

# Sidebar for CSV selection or upload
st.sidebar.markdown("### CSV Files")
use_uploaded_csv = st.sidebar.checkbox("Upload a new CSV file")
if use_uploaded_csv:
    uploaded_csv = st.sidebar.file_uploader("Upload your dataset (CSV)", type="csv")
    if uploaded_csv:
        csv_path = None
        data = pd.read_csv(uploaded_csv)
else:
    selected_csv_file = st.sidebar.selectbox("Select an example CSV file:", example_csv_files)
    csv_path = os.path.join(csv_folder, selected_csv_file)
    data = pd.read_csv(csv_path) if csv_path else None

# Sidebar for ML model selection or upload
st.sidebar.markdown("### ML Models")
use_uploaded_models = st.sidebar.checkbox("Upload new ML models (.pkl)")
selected_models = []

if use_uploaded_models:
    uploaded_models = st.sidebar.file_uploader("Upload your ML models (.pkl)", type="pkl", accept_multiple_files=True)
    if uploaded_models:
        selected_models = [load_pickle(model) for model in uploaded_models if model]
else:
    selected_example_models = st.sidebar.multiselect("Select example models:", example_pkl_files)
    selected_models = [load_pickle(os.path.join(pkl_folder, model)) for model in selected_example_models]

# Display dataset preview
if data is not None:
    st.write("### Dataset Preview")
    st.dataframe(data.head())

# Check if the `Dates` column exists
if data is not None and selected_models and "Dates" in data.columns:
    try:
        # Process data
        data['Dates'] = pd.to_datetime(data['Dates'], errors='coerce')
        data = data.dropna(subset=['Dates'])

        # Calculate rolling averages
        data['VIX_4Week_MA'] = data['VIX'].rolling(window=4).mean()
        data['DXY_4Week_MA'] = data['DXY'].rolling(window=4).mean()
        data['Cl1_4Week_MA'] = data['Cl1'].rolling(window=4).mean()
        data = data.dropna(subset=['VIX_4Week_MA', 'DXY_4Week_MA', 'Cl1_4Week_MA'])

        # Allow date selection
        available_dates = sorted(data['Dates'].drop_duplicates())
        selected_date = st.selectbox("Select a Date:", available_dates, format_func=lambda x: x.strftime("%Y-%m-%d"))

        # Filter for the selected date
        selected_row = data[data['Dates'] == pd.Timestamp(selected_date)]

        if not selected_row.empty:
            features = selected_row[['DXY_4Week_MA', 'VIX_4Week_MA', 'Cl1_4Week_MA']]

            # Collect probabilities from all models
            probabilities = {}
            for idx, model in enumerate(selected_models):
                model_name = f"Model {idx+1}" if isinstance(model, str) else f"Custom Model {idx+1}"
                try:
                    prob = model.predict_proba(features)[:, 1][0]  # Single prediction
                    probabilities[model_name] = prob
                except Exception as e:
                    st.warning(f"Error predicting with {model_name}: {e}")

            # Calculate average probability across all models
            avg_probability = sum(probabilities.values()) / len(probabilities)

            # Display results
            st.write(f"### Prediction Results for {selected_date}")
            st.plotly_chart(ut.create_gauge_chart(avg_probability), use_container_width=True)
            st.write(f"**Average Likelihood of Market Anomaly**: {avg_probability * 100:.2f}%")

            # Display individual model probabilities
            st.plotly_chart(ut.create_model_probability_chart(probabilities), use_container_width=True)

        else:
            st.warning("No data available for the selected date.")
    except Exception as e:
        st.error(f"Error during prediction workflow: {e}")
else:
    if data is None:
        st.warning("Please upload or select a CSV file.")
    elif not selected_models:
        st.warning("Please upload or select ML models (.pkl).")
    elif "Dates" not in data.columns:
        st.warning("The dataset must contain a 'Dates' column.")

# Final Notes
st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ using Streamlit!")
