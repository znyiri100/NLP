import streamlit as st
from IPython.core.interactiveshell import InteractiveShell
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, fcluster
import openai
import plotly.express as px
import json
import random
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

from sklearn.decomposition import PCA
#from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings

def name_clusters(clusters_df):
  ######################
  # rename cluster labels, give meaningful names to clusters
  ######################
  system_message = SystemMessage(
      content="You are an expert at naming clusters."
  )

  human_message = HumanMessagePromptTemplate.from_template(
      """You are a creative and insightful data scientist. Given a list of words, your task is to suggest three potential names that best describe the theme or relationship between the words.

      Here are the words: {cluster_words}

      Please provide your suggestions in the following JSON format:
      {{
        "cluster_label": {cluster_label}
        "cluster_names": ["name1", "name2", "name3"]
      }}"""
  )

  prompt = ChatPromptTemplate.from_messages([system_message, human_message])

  #llm = ChatOpenAI(model="o3-mini")
  llm = ChatOpenAI(model="gpt-4o-mini", temperature=1.0)
  #chain = LLMChain(llm=llm, prompt=prompt)
  chain = prompt | llm

  #st.write('Find name(s) for each cluster:')
  tmp_df = pd.DataFrame(); tmp_df2 = pd.DataFrame(); cluster_words_list=[]
  for label in clusters_df['label'].unique()[:]:
    cluster_words_list.append( {"cluster_words": ', '.join( clusters_df[clusters_df['label']==label]['word']), "cluster_label": f"{label}" } )
  #print(cluster_words_list)

  response = chain.batch( cluster_words_list )
  #print( response)

  for item in response:
    #print( item.content)
    try:
      # Parse the JSON string in the 'text' field
      parsed_data = json.loads(item.content.replace('```json','').replace('```',''))
      cluster_label = parsed_data['cluster_label']
      cluster_names = parsed_data['cluster_names']
      cluster_words = clusters_df[clusters_df['label'] == cluster_label]['word'].tolist()
      #print(f"Cluster {cluster_label}: {', '.join(cluster_names)}\n\t{cluster_words}")
      # Add words to df-s
      #tmp_df2 = pd.concat([tmp_df2, pd.DataFrame([{"label": cluster_label, "names": ', '.join(cluster_names), "words": ', '.join(cluster_words), "count": len(cluster_words) }])], ignore_index=True)
      new_data = pd.DataFrame( {"label": cluster_label, "name": [cluster_names[0] for word in cluster_words], "word": [word for word in cluster_words] })
      tmp_df = pd.concat([tmp_df, new_data], ignore_index=True)
    except json.JSONDecodeError as e:
      print(f"Error decoding JSON for item: {item}, Error: {e}")
  #st.write(tmp_df2)

  # add   embeddings
  tmp_df['word'] = tmp_df['word'].str.strip()
  # Merge the DataFrames, stripping whitespace during the merge
  tmp_df = pd.merge(tmp_df, clusters_df[['word', 'embeddings']],
                  # left_on=tmp_df['word'].str.strip(),  # Strip left DataFrame's 'word'
                  # right_on=clusters_df['word'].str.strip(),  # Strip right DataFrame's 'word'
                  on='word',
                  how='left')
  return tmp_df
    
def plot_3d(clusters_df, n=100, sample=True):
  """
  cluster_df: a Pandas DataFrame with labal, name, word, and optional embedding. Embedding is created if missing.
  n=100: The number of words to plot with default value of 100. It controls how many words are used for the plot.
  sample=True: If set to True, it indicates that the function should randomly select n words from the DataFrame. If set to False, it will use the first n words.
  """

  n=min(n, clusters_df.shape[0])

  if sample:
    tmp_df = clusters_df.sample(n=n).copy()
  else:
    tmp_df = clusters_df[:n].copy()

  if 'embeddings' in tmp_df.columns:
    print('We have embeddings, skip generation.')
    embeddings = tmp_df['embeddings'].tolist()
  else:
    print('Generate embeddings.')
    # Generate embeddings
    openai_embeddings=OpenAIEmbeddings()
    # Combine 'name' and 'word' into a single text field for embedding
    #tmp_df['combined_text'] = tmp_df.apply(lambda row: f"{row['name']}: {row['word']}", axis=1)
    tmp_df['combined_text'] = tmp_df.apply(lambda row: f"{row['word']}", axis=1)
    texts_to_embed = tmp_df['combined_text'].tolist()
    embeddings = openai_embeddings.embed_documents(texts_to_embed)

  #print( tmp_df.head(3), embeddings[:3] )

  # Reduce the dimensions using PCA (for visualization purposes)
  pca = PCA(n_components=3)
  embeddings_3d = pca.fit_transform(embeddings)

  # Create a DataFrame for Plotly
  df = pd.DataFrame(embeddings_3d, columns=['x', 'y', 'z'])
  df = df.reset_index(drop=True)
  tmp_df = tmp_df.reset_index(drop=True)
  df['word'] = tmp_df['word']  # Add 'word' column for labels
  df['name'] = tmp_df['name']  # Add 'name' column for coloring
  df['label'] = tmp_df['label']

  # Create a 3D scatter plot using Plotly with color based on 'name'
  fig = px.scatter_3d(
      df,
      x='x',
      y='y',
      z='z',
      text='word',  # Labels for the points
      color='name',  # Color points based on 'name' column
      title='3D PCA Visualization with Color by name',
  );

  # Update marker properties to make the dots smaller
  _ = fig.update_traces(
      marker=dict(
          size=5,         # Set marker size to a smaller value (e.g., 2)
          opacity=0.8,    # Optional: Adjust opacity for better visibility
          line=dict(width=0)  # Optional: Remove marker borders
      ),
      textposition='top center'  # Position text labels above the markers
  );

  # Update layout to increase plot size
  _ = fig.update_layout(
      autosize=False,  # Disable automatic sizing
      width=1000,    # Set the width of the plot (in pixels)
      height=800,   # Set the height of the plot (in pixels)
      margin=dict(l=5, r=5, b=5, t=5, pad=2),  # Adjust margins if needed
  );

  #fig.show()
  st.plotly_chart(fig)

def plot_2d(clusters_df, n=100, sample=True):

  # Shuffle the DataFrame using pandas' sample method
  clusters_df = clusters_df.sample(frac=1, random_state=42).reset_index(drop=True)

  # Select the max x points for plotting, otherwise too many points
  num_select = min(n, clusters_df.shape[0])

  pca = PCA(n_components=2)  # Reduce to 2D for visualization
  pca_result = pca.fit_transform( clusters_df['embeddings'].to_list() )

  # subset
  pca_result_subset = pca_result[:num_select]  # Subset of PCA results
  words_subset = clusters_df['word'][:num_select]  # Subset of words
  labels_subset = clusters_df['label'][:num_select]  # Subset of labels

  # Get unique cluster labels and assign colors
  unique_labels = np.unique(clusters_df['label'])
  colors = plt.get_cmap('viridis', len(unique_labels))  # Get a colormap with enough colors

  # Create scatter plot
  _ = plt.figure(figsize=(12, 9))
  _ = plt.scatter(pca_result_subset[:, 0],
                  pca_result_subset[:, 1],
                  c=labels_subset,
                  cmap=colors,  # Use the colormap directly
                  s=100)

  # Annotate each point with the word (using the subset)
  for i, word in enumerate(words_subset):
      label = labels_subset[i]  # Get the cluster label for the current word
      annotation_text = f" {label}-{word}"  # Combine word and label
      #annotation_text = f"{word}"  # Combine word and label
      _ = plt.annotate(annotation_text, (pca_result_subset[i, 0], pca_result_subset[i, 1]), fontsize=12)

  # Create legend elements
  legend_elements = []
  # iterate through the dataframe and create legend element for each cluster
  for label, name in clusters_df[['label','name']].drop_duplicates().values:
      # color is determined by the index of unique labels using the colormap
      color = colors(np.where(unique_labels == label)[0][0])
      legend_elements.append(mpatches.Patch(color=color, label=f'Cluster {label} - {name}'))

  # Add legend to the plot using the manually created elements
  _ = plt.legend(handles=legend_elements, loc='best')  # loc='best' finds the optimal position

  _ = plt.title(f'Clustering of Words with OpenAI Embeddings (First {num_select} Points)')
  _ = plt.xlabel('PCA Component 1')
  _ = plt.ylabel('PCA Component 2')
  #plt.show()
  st.pyplot(plt)

def plot_dendrogram(linkage_matrix, num_clusters=3):
  if linkage_matrix is None:
    print('linkage_matrix is None')
    return
  else:
    _=plt.figure(figsize=(10, 6))
    _=dendrogram(linkage_matrix, truncate_mode='lastp', p=num_clusters, orientation='left')
    _=plt.title(f'Hierarchical Clustering Dendrogram, {len(linkage_matrix)+1} data points')
    _=plt.xlabel('Number of points in group or the index of point if not in parenthesis')
    #_=plt.show()
    st.pyplot(plt)

def kmeans_clustering(word_list, num_clusters=3):
  # Your existing kmeans_clustering function

  # Retrieve embeddings from OpenAI (using the new method)
  response = openai.embeddings.create(
    #model="text-embedding-ada-002",  # Use the appropriate model
    model="text-embedding-3-small",
    input=word_list
  )

  # Get the embeddings from the response
  word_embeddings = [embedding.embedding for embedding in response.data]

  # Convert the embeddings to a numpy array for easier manipulation
  word_embeddings = np.array(word_embeddings)

  # Check the dimension of the embeddings
  print(f"OpenAI embeddings dimension: {word_embeddings.shape[1]}")

  # Step 3: Apply K-means clustering
  kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
  kmeans.fit(word_embeddings)

  # Assuming 'words' is the original list of words used for clustering
  clusters = {i: [] for i in range(kmeans.n_clusters)}  # Create empty lists for each cluster
  for i, word in enumerate(word_list):
      cluster_label = kmeans.labels_[i]  # Get the cluster label for the word
      clusters[cluster_label].append(word)  # Add the word to the corresponding cluster list

  # Create a list to store data for the DataFrame
  data = []
  for cluster_label, cluster_words in clusters.items():
      for word in cluster_words:
          data.append({'label': cluster_label, 'name': f'name{cluster_label}', 'word': word, 'embeddings': word_embeddings[word_list.index(word)]})

  # Create the DataFrame
  return pd.DataFrame(data).reset_index(drop=True)

def hierarchical_clustering(word_list=["apple", "computer", "banana", "laptop", "orange"], num_clusters=3):

    # Random word embeddings (you would typically use actual word embeddings here)
    #word_embeddings = np.random.rand(len(word_list), 50)

    # Retrieve embeddings from OpenAI
    response = openai.embeddings.create(
      #model="text-embedding-ada-002",  # Use the appropriate model
      model="text-embedding-3-small",
      input=word_list
    )
    word_embeddings = [embedding.embedding for embedding in response.data]
    word_embeddings = np.array(word_embeddings)

    # Perform AgglomerativeClustering without specifying n_clusters
    hierarchical_cluster = AgglomerativeClustering(linkage='ward', distance_threshold=0, n_clusters=None)

    # Fit the model to the word embeddings
    hierarchical_cluster.fit(word_embeddings)

    # counts for dendrogram
    counts = np.zeros(hierarchical_cluster.children_.shape[0])
    n_samples = len(hierarchical_cluster.labels_)
    for i, merge in enumerate(hierarchical_cluster.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    # Linkage matrix (this gives us the hierarchical structure and the distances)
    linkage_matrix = np.column_stack([hierarchical_cluster.children_, hierarchical_cluster.distances_, counts]).astype(float)
    #print(len(linkage_matrix))

    # The distances at the time the clusters were merged
    distances = linkage_matrix[-(n_samples - num_clusters):, 2]  # Last n_samples - num_clusters rows represent the merges at the final level
    #print(f"Distances for the merges when cutting at {num_clusters} clusters: {distances}")

    # Cut the tree to get the final cluster labels at the desired number of clusters
    final_labels = fcluster(linkage_matrix, t=num_clusters, criterion='maxclust')

    # Create DataFrame with the final cluster labels
    clusters_df = pd.DataFrame({
        'label': final_labels,
        'name': [f'name{label}' for label in final_labels],
        'word': word_list,
        'embeddings': [word_embeddings[i] for i in range(len(word_list))],
    })
    #print(clusters_df.head(1))
    return clusters_df, linkage_matrix