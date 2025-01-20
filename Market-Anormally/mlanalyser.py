import streamlit as st
import pickle
import os

# Streamlit app setup
st.title("Model Feature Analyzer 🧠📊")
st.markdown("Upload `.pkl` model files to analyze the features they require.")

# File uploader for .pkl files
uploaded_files = st.file_uploader(
    "Upload your `.pkl` files", type="pkl", accept_multiple_files=True
)

# Function to analyze a model
def analyze_model(file):
    try:
        # Load the model
        model = pickle.load(file)
        
        # Check the type of model and try to extract required features
        if hasattr(model, "get_booster") and hasattr(model.get_booster(), "feature_names"):
            # XGBoost models
            features = model.get_booster().feature_names
            model_type = "XGBoost"
        elif hasattr(model, "feature_importances_"):
            # Models with feature importances (e.g., Random Forest, Gradient Boosting)
            features = "Feature importances available (specific features not directly extractable)."
            model_type = type(model).__name__
        elif hasattr(model, "coef_"):
            # Logistic Regression
            features = "Coefficients available (specific features not directly extractable)."
            model_type = "Logistic Regression"
        elif hasattr(model, "support_"):
            # SVM
            features = "Feature support vector available (specific features not directly extractable)."
            model_type = "SVM"
        else:
            # Unsupported or unrecognized models
            features = "Unable to determine required features."
            model_type = type(model).__name__
        
        return {"model_type": model_type, "features": features}
    
    except Exception as e:
        return {"model_type": "Unknown", "features": f"Error analyzing model: {e}"}

# Process uploaded files
if uploaded_files:
    for uploaded_file in uploaded_files:
        st.write(f"### Analyzing: `{uploaded_file.name}`")
        
        # Analyze the model
        with uploaded_file:
            result = analyze_model(uploaded_file)
        
        # Display results
        st.write(f"**Model Type**: {result['model_type']}")
        if isinstance(result["features"], list):
            st.write("**Required Features**:")
            st.write(", ".join(result["features"]))
        else:
            st.write(f"**Features Information**: {result['features']}")
else:
    st.info("Upload `.pkl` files to analyze the required features.")

# Final message
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit!")
