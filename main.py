import streamlit as st
import os
#
import intro
import cluster
import classify
import topic_modeling

st.set_page_config(page_title="NLP Experiments", layout="wide")

# Check for OPENAI_API_KEY environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    openai_api_key = st.sidebar.text_input("Please enter your OpenAI API key:", type="password")
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key

# Sidebar for navigation
page = st.sidebar.radio("Pick something!", ["Intro", "Clustering", "Topic Modeling", "Classification"], index=0)
st.sidebar.markdown("---")  # This creates a horizontal line separator in Streamlit

# Navigation logic
if page == "Intro":
    intro.app()
elif page == "Clustering":
    cluster.app()
elif page == "Topic Modeling":
    topic_modeling.app()
elif page == "Classification":
    classify.app()