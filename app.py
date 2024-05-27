import streamlit as st
import numpy as np
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification

MODEL_FILE_DistilBERT = "/Users/chooenming/Downloads/DistilBert_Intermediate/Model/distilbert"
loaded_model_DistilBERT = TFDistilBertForSequenceClassification.from_pretrained(MODEL_FILE_DistilBERT)
MODEL_NAME = "distilbert-base-uncased"
loaded_tokenizer_DistilBERT = DistilBertTokenizer.from_pretrained(MODEL_NAME)

def predict_suicide(statement):
    inputs = loaded_tokenizer_DistilBERT(statement, truncation=True, padding="max_length", return_tensors="tf")
    prediction = loaded_model_DistilBERT.predict([inputs["input_ids"], inputs["attention_mask"]])
    prediction_proba = prediction.logits[:, 0]
    prediction_probabilities = tf.nn.sigmoid(prediction_proba).numpy()
    prediction_class = (prediction_probabilities>0.5).astype(int)

    return prediction_class[0]

# set the page title and favicon
st.set_page_config(page_title="Suicide Ideation Detection Chatbot",
                   page_icon=":broken_heart:")

# chatbot theme customisation
st.markdown(
    """
    <style>
    .chat-container {
        background-color: #f0f0f0;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
        max-width: 600px;
    }
    .user-message{
        background-color: #e5f2ff;
        padding: 10px 15px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .bot-message{
        background-color: #b3d9ff;
        padding: 10px 15px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .footer{
        font-size: 12px;
        text-align: center;
        padding-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html = True
)

# Header
st.title("Suicide Ideation Detection Chatbot")
st.markdown("---")

chat_container = st.container()
with chat_container:
    st.markdown("**Welcome! Try ME!**")
    st.text("")
    st.markdown("Type a statement and I will predict if there is any potential of suicide risk.")
    st.text("")
    

# main content
with st.container():
    statement = st.text_input("You: ", "")
    if statement:
        with chat_container:
            st.text("")
            st.markdown(f"**You:** {statement}", unsafe_allow_html=True)
            st.text("")

            # make prediction
            prediction = predict_suicide(statement)
            result_text = "Potential of Suicide" if prediction == 1 else "No Potential of Suicide"
            st.markdown(f"**Chatbot:** <b>{result_text}</b> ", unsafe_allow_html=True)
            st.text("")
            if prediction == 1:
                st.markdown("Please seek help from the counselor.")
                st.text("")
                st.markdown("**Your life is precious! ❤️**")
            else:
                st.markdown("Have a nice day! 😊")

# footer
st.markdown("---")
st.markdown("Made with ❤️❤️❤️ by Choo En Ming")
