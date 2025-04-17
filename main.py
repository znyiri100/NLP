import streamlit as st
import os

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

# can run from within same folder or in nlp folder
folder = 'nlp' if os.path.exists('nlp') else '.'

# Navigation logic
if page == "Intro":
    exec(open(f"{folder}/intro.py").read())
elif page == "Clustering":
    exec(open(f"{folder}/cluster.py").read())
elif page == "Topic Modeling":
    exec(open(f"{folder}/topic_modeling.py").read())
elif page == "Classification":
    exec(open(f"{folder}/classify.py").read())