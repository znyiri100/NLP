# prompt: Using the Python library Streamlit, construct an interactive web dashboard that either ask for word_list, or ask for a file like word_list_B1.csv to load the words from.
# Classify the given list using kmeans or hierarchical methods selectable by checkmarks. Ask for the num_clusters.
# Rename the cluster labels using name_clusters() function.
# Calculate the number of words in each group.
# Visualizes the word embeddings in 2D or 3D space. The graphic type is selectable by chakmarks.
# In case of hierarchical method dendogram also possible to select.
import streamlit as st
from IPython.core.interactiveshell import InteractiveShell
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, fcluster
import openai
import plotly.express as px
#import matplotlib.pyplot as plt # Not used, remove if not needed
import os

from cluster_helper import name_clusters, plot_3d, plot_2d, plot_dendrogram, kmeans_clustering, hierarchical_clustering
from nlp_utils import f_word_list

#def app():
st.title("Cluster Words")

# read in words
word_list_df = f_word_list()
if word_list_df.shape[0]>0:
    word_list_input = st.text_area("Enter words (comma-separated or one per line):", value='\n'.join(word_list_df['word'].tolist()) )
    if word_list_input:
        word_list = [word.strip() for word in word_list_input.replace(',', '\n').splitlines() if word.strip()]
    st.write(f"{len(word_list)} words.")

# Clustering method selection
clustering_method = st.sidebar.selectbox("Select Clustering Method:", ("KMeans", "Hierarchical"))

num_clusters = st.sidebar.number_input("Number of Clusters:", min_value=2, value=3)

# OpenAI API Key
#openai.api_key = st.secrets["OPENAI_API_KEY"] # Get the API key from Streamlit secrets

# Visualizations
show_2d_plot = st.sidebar.checkbox("Show 2D Plot")
show_3d_plot = st.sidebar.checkbox("Show 3D Plot")
if clustering_method == "Hierarchical":
    show_dendrogram = st.sidebar.checkbox("Show Dendrogram")
        
# Run the selected clustering method
if st.sidebar.button("Let's do it!"):
    if word_list:  # Check if the word list is not empty
        if clustering_method == "KMeans":
            clusters_df = kmeans_clustering(word_list, num_clusters)
        else:
            clusters_df, linkage_matrix = hierarchical_clustering(word_list, num_clusters)

            # Display the dendrogram (if hierarchical clustering is selected)
            if show_dendrogram:
                plot_dendrogram(linkage_matrix, num_clusters)

        clusters_df = name_clusters(clusters_df)
        #st.write(f'{clusters_df.shape[0]} words, sample rows:')
        #st.dataframe(clusters_df[['label', 'name', 'word']].head(3))
        
        cluster_counts = clusters_df.groupby(['label','name'])['word'].count().reset_index()
        st.write("Clusters:") 
        st.write(clusters_df.groupby(['label', 'name']).agg( count=('word', 'size'), words=('word', lambda x: list(x)) ).sort_values('count', ascending=False))
        st.write("Word list:") 
        st.write(clusters_df[['word', 'name']])

        # Visualizations
        if show_2d_plot:
            plot_2d(clusters_df)
        if show_3d_plot:
            plot_3d(clusters_df)

    else:
        st.warning("Please enter a word list or upload a CSV file first.")
