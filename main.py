import streamlit as st
import intro
import cluster
import classify
import topic_modeling

st.set_page_config(page_title="NLP Experiments", layout="wide")

# Sidebar for navigation
#st.sidebar.title("Navigation")
page = st.sidebar.radio("Pick something!", ["Intro", "Clustering", "Topic Modeling", "Classification"], index=2)
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