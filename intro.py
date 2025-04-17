import streamlit as st

#def app():
st.markdown("""<h3>NLP methods</h3>
    <strong>Clustering, Topic Modeling, Classification, and Named Entity Recognition (NER)</strong> are all key natural language processing (NLP) techniques, but they serve different purposes.<br><br>Here's a breakdown of each:
<table>
    <tr>
        <th>Method</th>
        <th>Goal</th>
        <th>Common Algorithms</th>
        <th>Use Cases</th>
    </tr>
    <tr>
        <td>Clustering</td>
        <td>Group similar documents or data points together.</td>
        <td>K-means, DBSCAN, Hierarchical Clustering, GMM</td>
        <td>Grouping documents, identifying customer segments, detecting anomalies</td>
    </tr>
    <tr>
        <td>Topic Modeling</td>
        <td>Identify topics that appear in a collection of text documents.</td>
        <td>LDA, LSA, NMF</td>
        <td>Summarizing text data, organizing documents, identifying trends</td>
    </tr>
    <tr>
        <td>Classification</td>
        <td>Assign predefined labels or categories to a given text.</td>
        <td>Naive Bayes, SVM, Neural Networks, BERT, Random Forests</td>
        <td>Sentiment analysis, spam detection, document categorization, customer feedback analysis</td>
    </tr>
    <tr>
        <td>Named Entity Recognition (NER)</td>
        <td>Identify and classify named entities in text.</td>
        <td>CRFs, SpaCy, BERT-based models</td>
        <td>Extracting structured information, enhancing search engines, data extraction</td>
    </tr>
</table>

<h3>The challenge</h3>
While developing the curriculum for our AI-driven language learning app, Japi (https://japi.ai/), we had to categorize thousands of words from the CEFR Oxford Learner's Word Lists (https://www.oxfordlearnersdictionaries.com/us/wordlists/).
<br>Traditional classification methods alone were insufficient due to the lack of context—we were dealing with standalone words. However, pretrained LLMs possess contextual knowledge of words. Therefore, we leveraged LLMs to generate themes and classify the words accordingly.
<br>An additional challenge was ensuring that the topics we assigned to words aligned with the thematic units in our curriculum. This was overcome through precise prompt engineering.

<h3>Solutions</h3>
Recent advancements in LLMs enabled new ways to analyse text. We are constructing a reliable text classification pipeline using LLMs, emphasizing techniques like constrained generation, and few-shot prompting to enhance accuracy. The core idea is to democratize and streamline traditionally complex NLP tasks, making them accessible to a broader audience through LLMs.
<br><br>One method we tested began with traditional machine learning models. We found that both K-Means and Hierarchical Clustering were fast and effective in grouping similar words when fed word embeddings. However, cluster labels were mere numbers—far from ideal for integrating into our curriculum. To resolve this, we used LLMs and dynamic one-shot prompting to generate meaningful, user-friendly names for each cluster. The combination of traditional ML and this novel LLM-based approach proved highly effective.
<br><br>In another approach—rather than using traditional topic modeling techniques—we employed LLMs to generate three possible topics for each word. With a subsequent prompt, we consolidated these topics into a set number of main themes. Finally, using an LLM once more, we reassigned words to the most suitable main topic.
<br><br>The combination of these methods enabled us to categorize our word lists into well-aligned thematic units, enhancing the overall structure of our curriculum.
<h3>Implementation</h3>
<strong>Clustering:</strong> This script is a Streamlit application designed for clustering words using either KMeans or Hierarchical clustering methods. It allows users to input words through a text area or by uploading a file. Then--using an LLM call--it generates meanigful names for the clusters. The application provides options for visualizing the clustering results in 2D and 3D plots, as well as displaying a dendrogram for hierarchical clustering.
<br><br><strong>Topic Modeling:</strong> This is a Streamlit application for topic modeling, which categorizes a list of words into main topics using the LangChain library and OpenAI's GPT model. The script takes a list of words and generates topics by sending each word to a language model to get potential topic categories. It is consolidating these categories into a specified number of main topics.
<br><br><strong>Classification:</strong> This is a Streamlit application designed for classifying words into predefined topics using a language model. A set of topics is defined within the code, which the application uses for categorization. The app utilizes the LangChain library and OpenAI's GPT model to analyze each word and assign it to one of the specified topics.
<h4>Summary</h4>

- **Clustering Application:**
  - Allows users to input words manually or upload a file.
  - Groups words using K-Means or Hierarchical Clustering.
  - Renames the numeric cluster labels to more meaningful names using AI.
  - Using 3D and 2D scatter plot to see how the groups of words are placed in the PCA-reduced feature space.
  - Create dendrogram to illustrate word groupings for hierarchical clustering.
<br><br>

- **Topic Modeling Application:**
  - Utilizes the LangChain library and OpenAI's GPT model.
  - Takes a list of words and prompts the AI to suggest topics.
  - Consolidates the suggestions into a specified number of main topics.
<br><br>

- **Classification Application:**
  - Also employs the LangChain library and GPT model.
  - Accepts a list of words along with a set of predefined topics.
  - Analyzes each word and assigns it to one of the specified topics.
  - Using 3D and 2D scatter plot to see how the groups of words are placed in the PCA-reduced feature space.
""", unsafe_allow_html=True)