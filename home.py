import streamlit as st

def app():
    st.markdown("<strong>Topic Modeling, Named Entity Recognition (NER), Text Classification, and Clustering</strong> are all key natural language processing (NLP) techniques, but they serve different purposes.<br><br>Here's a breakdown of each:", unsafe_allow_html=True)
    st.markdown("""
    <h3>1. Topic Modeling</h3>
    <p><strong>Goal:</strong> Identify topics that appear in a collection of text documents.</p>
    <ul>
        <li><strong>Definition:</strong> Topic modeling is a technique used to discover the underlying themes or topics that are present in a corpus of text data. It’s an unsupervised learning method, meaning it doesn’t rely on pre-labeled data.</li>
        <li><strong>Output:</strong> A set of topics, where each topic is represented as a collection of words that frequently occur together. Each document in the corpus is then associated with one or more of these topics.</li>
        <li><strong>Common Algorithms:</strong> Latent Dirichlet Allocation (LDA), Latent Semantic Analysis (LSA), Non-Negative Matrix Factorization (NMF).</li>
        <li><strong>Use Cases:</strong>
            <ul>
                <li>Summarizing large volumes of text data.</li>
                <li>Organizing documents into topic clusters.</li>
                <li>Identifying emerging trends in data, such as in social media or news articles.</li>
            </ul>
        </li>
    </ul>
    
    <h3>2. Named Entity Recognition (NER)</h3>
    <p><strong>Goal:</strong> Identify and classify named entities (e.g., people, organizations, locations) in text.</p>
    <ul>
        <li><strong>Definition:</strong> NER is a process in which entities like names of people, organizations, dates, locations, etc., are identified and categorized in text. It’s a subtask of information extraction.</li>
        <li><strong>Output:</strong> A list of entities (e.g., “Barack Obama” → PERSON, “New York” → LOCATION) in the text.</li>
        <li><strong>Common Algorithms:</strong> Conditional Random Fields (CRFs), SpaCy, BERT-based models, and other deep learning models.</li>
        <li><strong>Use Cases:</strong>
            <ul>
                <li>Extracting structured information from unstructured text.</li>
                <li>Enhancing search engines with entity-focused queries.</li>
                <li>Data extraction for applications like customer service (e.g., pulling out names, places from emails).</li>
            </ul>
        </li>
    </ul>
    
    <h3>3. Text Classification</h3>
    <p><strong>Goal:</strong> Assign predefined labels or categories to a given text.</p>
    <ul>
        <li><strong>Definition:</strong> Text classification involves categorizing a piece of text into predefined categories or classes, such as spam vs. not spam, sentiment analysis (positive, negative, neutral), or topic classification (sports, politics, etc.). This is typically a supervised learning task.</li>
        <li><strong>Output:</strong> A label or category assigned to the text. For example, classifying a news article as “Technology” or “Health.”</li>
        <li><strong>Common Algorithms:</strong> Naive Bayes, SVM (Support Vector Machines), Neural Networks, BERT, Random Forests.</li>
        <li><strong>Use Cases:</strong>
            <ul>
                <li>Sentiment analysis (positive, negative, neutral).</li>
                <li>Spam email detection.</li>
                <li>Categorizing documents or articles (e.g., classifying news articles into categories like sports, politics, entertainment).</li>
                <li>Customer feedback analysis.</li>
            </ul>
        </li>
    </ul>
    
    <h3>4. Clustering</h3>
    <p><strong>Goal:</strong> Group similar documents or data points together.</p>
    <ul>
        <li><strong>Definition:</strong> Clustering is an unsupervised learning technique where similar data points are grouped into clusters. Unlike topic modeling, which identifies latent topics in a set of documents, clustering groups documents based on their similarity. It’s often used when the true groupings in the data are unknown.</li>
        <li><strong>Output:</strong> A set of clusters, where each cluster contains similar items (e.g., documents or data points).</li>
        <li><strong>Common Algorithms:</strong> K-means, DBSCAN (Density-Based Spatial Clustering of Applications with Noise), Hierarchical Clustering, and Gaussian Mixture Models (GMM).</li>
        <li><strong>Use Cases:</strong>
            <ul>
                <li>Grouping documents by similarity (e.g., categorizing news articles without predefined labels).</li>
                <li>Identifying customer segments for marketing.</li>
                <li>Detecting anomalies or outliers in data.</li>
                <li>Organizing large datasets or helping with exploratory data analysis.</li>
            </ul>
        </li>
    </ul>
    """, unsafe_allow_html=True)
    