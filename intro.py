import streamlit as st

def app():
    st.markdown("<h3>NLP methods</h3>", unsafe_allow_html=True)
    st.markdown("<strong>Clustering, Topic Modeling, Classification, and Named Entity Recognition (NER)</strong> are all key natural language processing (NLP) techniques, but they serve different purposes.<br><br>Here's a breakdown of each:", unsafe_allow_html=True)

    st.markdown("""
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
    """, unsafe_allow_html=True)

    st.markdown("<h3>Our implementation:</h3>", unsafe_allow_html=True)
    st.markdown("Recent advancements in LLMs enabled new ways to analyse text. We are constructing a reliable text classification pipeline using LLMs, emphasizing techniques like constrained generation, few-shot prompting, and dynamic example selection to enhance accuracy. The core idea is to democratize and streamline traditionally complex NLP tasks, making them accessible to a broader audience through LLMs.", unsafe_allow_html=True)
    st.markdown("**Clustering**: This script is a Streamlit application designed for clustering words using either KMeans or Hierarchical clustering methods. It allows users to input words through a text area or by uploading a file. The application provides options for visualizing the clustering results in 2D and 3D plots, as well as displaying a dendrogram for hierarchical clustering.", unsafe_allow_html=True)
    st.markdown("**Topic Modeling**: This is a Streamlit application for topic modeling, which categorizes a list of words into main topics using the LangChain library and OpenAI's GPT model. The script takes a list of words and generates topics by sending each word to a language model to get potential topic categories. It is consolidating these categories into a specified number of main topics.", unsafe_allow_html=True)
    st.markdown("**Classification**: This is a Streamlit application designed for classifying words into predefined topics using a language model. A set of topics is defined within the code, which the application uses for categorization. The app utilizes the LangChain library and OpenAI's GPT model to analyze each word and assign it to one of the specified topics.", unsafe_allow_html=True)