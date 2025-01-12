import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
from openai import OpenAI
import utils as ut

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ['GROQ_API_KEY'],
)


def load_model(filename):
  with open(filename, 'rb') as file:
    return pickle.load(file)


xgboost_model = load_model('xgb_model.pkl')

naive_baye_model = load_model('mb_model.pkl')

svm_model = load_model('svm_model.pkl')

def prepare_input(df):
    df = pd.read_csv('FinancialMarketData.xlsx - EWS.csv')
    df['Data'] = pd.to_datetime(df['Data'])

    df['VIX_4Week_MA'] = df['VIX'].rolling(window=4).mean()
    df['DXY_4Week_MA'] = df['DXY'].rolling(window=4).mean()
    df['Cl1_4Week_MA'] = df['Cl1'].rolling(window=4).mean()
    input_dict = {'VIX_4Week_MA': df['VIX_4Week_MA'][1],
        'DXY_4Week_MA': df['DXY_4Week_MA'][1],
        'Cl1_4Week_MA': df['Cl1_4Week_MA'][1]
     }
    
    selected_columns = ['DXY_4Week_MA', 'VIX_4Week_MA', 'Cl1_4Week_MA']
    features = df[selected_columns]
    return features,input_dict


def make_predictions(input_df, input_dict):
    # Reorder input_df to match the feature order the model expects
    selected_columns = ['DXY_4Week_MA', 'VIX_4Week_MA', 'Cl1_4Week_MA']
    features = input_df[selected_columns]
    probabilities = {
        'Random forest feature engineered':
        xgboost_model.predict_proba(features)[0][1],
        'XGBOOST SMOTE':
        svm_model.predict_proba(features)[0][1],
        'XGBOOST feature engineered':
        naive_baye_model.predict_proba(features)[0][1],
    }

    avg_probability = np.mean(list(probabilities.values()))

    col1, col2 = st.columns(2)

    with col1:
        fig = ut.create_gauge_chart(avg_probability)
        st.plotly_chart(fig, use_container_width=True)
        st.write(
            f"There is a {avg_probability * 100:.2f}% chance of a market anormaly."
        )

    with col2:
        fig = ut.create_model_probability_chart(probabilities)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Anormaly Probabilities")
    for model, prob in probabilities.items():
        st.write(f"{model} {prob}")
    st.write(f"Average Probability: {avg_probability}")
    return avg_probability


def explain_prediction(probability):
    prompt = f"""You are an expert data scientist at a financial trading firm, specializing in interpreting and explaining predictions of machine learning models related to Anormalies in the stock market.

Your machine learning model has predicted that an anormally has a {round(probability * 100, 1)}% probability of churning, based on the information provided below.

Here is the Anormally information:
{input_dict}


{pd.set_option('display.max.columns', None)}

Below are the summary statistics for anormallies:
{df[df['Y'] == 1].describe()}

Below are the summary statistics for not market anormallies:
{df[df['Y'] == 0].describe()}

### Instructions:
1. Provide an investment stradegy on what stocks to and not to buy

2. Give a reason for your investment strategy


Now generate your response following the structure and guidelines provided.

"""

    print("EXPLANATION PROMPT", prompt)

    raw_response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{
            "role": "user",
            "content": prompt
        }],
    )
    return raw_response.choices[0].message.content


def generate_email(probability, input_dict, explanation, surname):
  prompt = f"""You are a manager at HS Bank. You are responsible for
ensuring customers stay with the bank and are incentivized with
various offers.

You noticed a customer named {surname} has a {round(probability * 
100, 1)}% probability of churning.

Here is the customer's information:
{input_dict}

Here is some explanation as to why the customer might be at risk 
of churning:
{explanation}

Generate an email to the customer based on their information, 
asking them to stay if they are at risk of churning, or offering them 
incentives so that they become more loyal to the bank.

Make sure to list out a set of incentives to stay based on their 
information, in bullet point format. Don't ever mention the 
probability of churning, or the machine learning model to the 
customer.
"""
  raw_response = client.chat.completions.create(model="Llama-3.1-8b-instant",
                                                messages=[{
                                                    "role": "user",
                                                    "content": prompt
                                                }])

  print("\\n\\nEMAIL PROMPT", prompt)

  return raw_response.choices[0].message.content


st.title("Stock Market Anormally predictor")

df = pd.read_csv("FinancialMarketData.xlsx - EWS.csv")

input_df, input_dict = prepare_input(df)
avg_probability = make_predictions(input_df,input_dict)

explanation = explain_prediction(avg_probability, input_df)

st.markdown("---")

st.subheader("Explanation of Prediction")

st.markdown(explanation)



st.markdown("---")

st.subheader("Personalized Email")
