import plotly.express as px
import streamlit as st
import pandas as pd
#from langchain.embeddings.openai import OpenAIEmbeddings
#from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from sklearn.decomposition import PCA

def create_category_distribution_chart(category_counts):
    """Create and display a bar chart showing category distribution."""
    fig = px.bar(
        x=category_counts.index,
        y=category_counts.values,
        labels={'x': 'Class', 'y': 'Count'},
        text=category_counts.values
    )
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

def create_embedding_projector(results_df):
    """Create and display 3D embedding projector."""
    #st.write("3D Embedding Projector")

    # Get top 10 categories
    top_10_categories = results_df['topic'].value_counts().nlargest(10).index
    filtered_df = results_df[results_df['topic'].isin(top_10_categories)].copy()

    openai_embeddings = OpenAIEmbeddings()
    embeddings = []
    progress_bar = st.progress(0)
    total_words = len(filtered_df)

    for i, word in enumerate(filtered_df['word']):
        embedding = openai_embeddings.embed_query(word)
        embeddings.append(embedding)
        progress_bar.progress((i + 1) / total_words)

    filtered_df['embedding'] = embeddings

    # Reduce the dimensions using PCA
    pca = PCA(n_components=3)
    embeddings_3d = pca.fit_transform(embeddings)

    # Create a DataFrame for Plotly
    df = pd.DataFrame(embeddings_3d, columns=['x', 'y', 'z'])
    df = df.reset_index(drop=True)
    filtered_df = filtered_df.reset_index(drop=True)
    df['word'] = filtered_df['word']
    df['topic'] = filtered_df['topic']
    #st.write(df.head(3))

    # Create 3D scatter plot
    fig = px.scatter_3d(
        df,
        x='x',
        y='y',
        z='z',
        text='word',
        color='topic'
    )
    # Update marker properties to make the dots smaller
    fig.update_traces(
        marker=dict(
            size=7,         # Set marker size to a smaller value (e.g., 2)
            opacity=0.8,    # Optional: Adjust opacity for better visibility
            line=dict(width=0)  # Optional: Remove marker borders
        ),
        textposition='top center'  # Position text labels above the markers
    )
    fig.update_layout(
        title={
            'text': '3D PCA Visualization of Top 10 Topics',
            'x': 0.5,  # Center the title horizontally
            'xanchor': 'center',
            'y': 1  # Move the title lower
        }
    )

    # Update layout to increase plot size
    fig.update_layout(
        autosize=False,  # Disable automatic sizing
        width=1000,    # Set the width of the plot (in pixels)
        height=800,   # Set the height of the plot (in pixels)
        margin=dict(l=5, r=5, b=5, t=5, pad=2),  # Adjust margins if needed
    );

    st.plotly_chart(fig, use_container_width=True)

