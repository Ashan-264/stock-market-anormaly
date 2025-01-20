import streamlit as st
import pandas as pd
import pickle
import utils as ut

# Helper function to load .pkl files
def load_pickle(file):
    try:
        return pickle.load(file)
    except Exception as e:
        st.error(f"Error loading the file: {e}")
        return None

# Streamlit App
st.title("Dynamic Stock Market Predictor 🧠📈")
st.sidebar.title("Configuration")
st.sidebar.markdown("Upload your models or datasets here!")

# Upload and manage pickle files
st.sidebar.subheader("Upload Models")
uploaded_files = st.sidebar.file_uploader(
    "Upload your `.pkl` models", type="pkl", accept_multiple_files=True
)

if uploaded_files:
    st.sidebar.success(f"{len(uploaded_files)} file(s) uploaded successfully!")
    models = {}
    for uploaded_file in uploaded_files:
        model_name = uploaded_file.name.split(".")[0]
        models[model_name] = load_pickle(uploaded_file)

    st.sidebar.markdown("### Available Models:")
    for model_name in models:
        st.sidebar.write(f"- {model_name}")

# Dataset Upload
st.sidebar.subheader("Upload Dataset")
uploaded_data = st.sidebar.file_uploader("Upload your dataset (CSV)", type="csv")

if uploaded_data:
    try:
        # Load dataset
        data = pd.read_csv(uploaded_data)

        # Extract dates from the `Dates` column
        data['Dates'] = pd.to_datetime(data['Dates'], errors='coerce')

        # Drop rows with invalid or missing dates
        data = data.dropna(subset=['Dates'])

        # Calculate rolling averages (4-week moving averages)
        data['VIX_4Week_MA'] = data['VIX'].rolling(window=4).mean()
        data['DXY_4Week_MA'] = data['DXY'].rolling(window=4).mean()
        data['Cl1_4Week_MA'] = data['Cl1'].rolling(window=4).mean()

        # Drop rows with missing rolling averages
        data = data.dropna(subset=['VIX_4Week_MA', 'DXY_4Week_MA', 'Cl1_4Week_MA'])

        # Display dataset preview
        st.write("### Dataset Preview")
        st.dataframe(data.head())

        # Extract unique dates from the `Dates` column
        available_dates = sorted(data['Dates'].drop_duplicates())
    except Exception as e:
        st.error(f"Error processing dataset: {e}")
else:
    available_dates = []

# Select a model to use
if uploaded_files:
    selected_model = st.selectbox("Select a model for predictions:", list(models.keys()))
else:
    selected_model = None

# Prediction workflow
if uploaded_files and available_dates:
    # Allow user to select a date from the `Dates` column
    selected_date = st.selectbox("Select a Date from the dataset:", available_dates, format_func=lambda x: x.strftime("%Y-%m-%d"))
    st.write(f"Selected Date: {selected_date}")

    # Filter the dataset for the selected date
    selected_row = data[data['Dates'] == pd.Timestamp(selected_date)]

    if not selected_row.empty:
        try:
            # Extract features for the models
            features = selected_row[['DXY_4Week_MA', 'VIX_4Week_MA', 'Cl1_4Week_MA']]

            # Collect probabilities from all models
            probabilities = {}
            for model_name, model in models.items():
                prob = model.predict_proba(features)[:, 1][0]  # Single prediction
                probabilities[model_name] = prob

            # Calculate average probability across all models
            avg_probability = sum(probabilities.values()) / len(probabilities)

            # Display results
            st.write(f"### Prediction Results for {selected_date}")
            st.plotly_chart(ut.create_gauge_chart(avg_probability), use_container_width=True)
            st.write(f"**Average Likelihood of Market Anomaly**: {avg_probability * 100:.2f}%")

            # Display individual model probabilities
            st.plotly_chart(ut.create_model_probability_chart(probabilities), use_container_width=True)

        except KeyError as e:
            st.error(f"Missing required columns for prediction: {e}")
        except Exception as e:
            st.error(f"Error preparing features or making predictions: {e}")
    else:
        st.warning("No data available for the selected date.")

# Final Notes
st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ using Streamlit!")
