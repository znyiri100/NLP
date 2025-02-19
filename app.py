import streamlit as st
import home
import cluster
import classify

# Sidebar for navigation
#st.sidebar.title("Navigation")
page = st.sidebar.radio("Pick something!", ["Home", "Cluster", "Classify"], index=0)

# Navigation logic
if page == "Home":
    home.app()
if page == "Cluster":
    cluster.app()
elif page == "Classify":
    classify.app()