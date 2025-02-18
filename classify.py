import streamlit as st
import pandas as pd
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
import os
import time
from datetime import datetime

from classify_charts import create_category_distribution_chart, create_embedding_projector

# Initialize results_df in session state if it doesn't exist
if 'results_df' not in st.session_state:
    st.session_state.results_df = None

# Streamlit config and title
st.set_page_config(page_title="Word Classifier", layout="wide")
st.title("Word Classification")

#api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
# api_key = 'sk-...'
# if api_key:
#     os.environ["OPENAI_API_KEY"] = api_key

# Define topics
topics = '''Travel and Transportation
Shopping and Money
Work and Professions
Hobbies
Environment and Weather
Education
Learning
Technology and Media
Food and Cooking
Housing and Furniture
Family and Traditions
Relationships
People
Behavior
Emotions
Interaction
Health
Fitness
'''

# Define the output schema
class WordCategory(BaseModel):
    word: str = Field(description="The input word")
    topic: str = Field(description="The assigned topic category")

# Create parser
parser = PydanticOutputParser(pydantic_object=WordCategory)

# file upload to sidebar
with st.sidebar:
    #st.title("Configuration")

    # File upload
    uploaded_file = st.file_uploader("Upload your word list (txt file)", type="txt")

if uploaded_file:
    # Read the uploaded file
    df = pd.read_csv(uploaded_file, header=None, names=['word'])

    # Add word selection options in sidebar
    st.sidebar.subheader("Word Selection")
    selection_method = st.sidebar.radio(
        "Select words by:",
        ["First N words", "Random sample"]
    )
    
    # Add slider for number of words to process
    n_words = st.sidebar.slider(
        "Number of words to process",
        min_value=1,
        max_value=df.shape[0],
        value=min(10, df.shape[0])
    )
    
    # Process words based on selection method
    if selection_method == "Random sample":
        df2 = df.sample(n=n_words, random_state=42)
    else:
        df2 = df.head(n_words)

if uploaded_file:
    
    # Create two columns for words and topics
    col1, col2 = st.columns(2)
    
    # Column 1: Display the words
    with col1:
        editable_words = st.text_area("Edit words (one per line):", value='\n'.join(df2['word'].tolist()))
        df2 = pd.DataFrame({'word': editable_words.split('\n')})
        st.write(f"{df2.shape[0]} words")
    
    # Column 2: Display and edit topics  
    with col2:
        topics_list = topics.strip().split('\n')
        editable_topics = st.text_area("Edit classes (one per line):", value=topics)
        topics_list = editable_topics.strip().split('\n')
        st.write(f"{len(topics_list)} classes")

    # Button to start categorization
    if st.button("Classify Words"):
        
        try:
            # Initialize progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Initialize LLM
            llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
            
            system_message = SystemMessage(
                content="You are an expert at categorizing words. Assign words to one of the topics it belongs to the most."
            )
            
            # Create prompt with format instructions
            human_message = HumanMessagePromptTemplate.from_template(
                """Analyze this word and assign to one of the topics:
                word: {word}
                topics: {topics}
                
                {format_instructions}"""
            )
            
            # Update prompt to include format instructions
            prompt = ChatPromptTemplate.from_messages([
                system_message,
                human_message
            ]).partial(format_instructions=parser.get_format_instructions())
            
            # Create chain
            chain = prompt | llm | parser
            
            # Prepare batch inputs
            word_list = df2.word.tolist()
            inputs = [{"word": word, "topics": editable_topics} for word in word_list]
            
            # Process in batches
            batch_size = max(df2.shape[0] // 10, 20)  # Ensure minimum batch size of 10
            results = []
            
            start_time = time.time()
            for i in range(0, len(inputs), batch_size):
                batch = inputs[i:i + batch_size]
                batch_results = chain.batch(inputs=batch)
                results.extend(batch_results)
                
                # Update progress
                progress = (i + len(batch)) / len(inputs)
                progress_bar.progress(progress)
                status_text.text(f"Processing words {i + 1} to {min(i + batch_size, len(inputs))} of {len(inputs)}")
            
            st.write(f"Execution time: {time.time() - start_time:.1f}s, batch size: {batch_size}")
            
            # Create results DataFrame
            results_df = pd.DataFrame([
                {"word": item.word, "topic": item.topic} 
                for item in results
            ])
            
            # Store results in session state
            st.session_state.results_df = results_df

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            
# elif not api_key:
#     st.warning("Please enter your OpenAI API key in the sidebar.")
elif not uploaded_file:
    st.info("Please upload a text file containing your word list.")

def toggle_chart():
    st.session_state.show_chart = not st.session_state.show_chart

def toggle_chart2():
    st.session_state.show_chart2 = not st.session_state.show_chart2

# Initialize the states if they don't exist
if 'show_chart' not in st.session_state:
    st.session_state.show_chart = False
if 'show_chart2' not in st.session_state:
    st.session_state.show_chart2 = False

# Create checkboxes with on_change handlers
st.sidebar.checkbox(
    "Category Distribution Chart",
    value=st.session_state.show_chart,
    on_change=toggle_chart
)

st.sidebar.checkbox(
    "3D Embedding Projector",
    value=st.session_state.show_chart2,
    on_change=toggle_chart2
)

if (uploaded_file and st.session_state.results_df is not None):

    # Create three columns for displaying results
    col_counts, col_results, col_download = st.columns(3)
    
    # Display category counts in left column
    with col_counts:
        with st.expander("Category Counts", expanded=False):
            #st.subheader(f"Category Counts for {results_df.shape[0]} words")
            category_counts = st.session_state.results_df['topic'].value_counts()
            st.write(category_counts)
    
    # Display results in middle column
    with col_results:
        with st.expander("Classification Results", expanded=False):
            st.subheader("Classification Results")
            st.write(st.session_state.results_df)

    # Display download button in right column
    with col_download:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"classified_words_{timestamp}.csv"
        #st.subheader("Download Results")
        st.download_button(
            label=f"Download as CSV",
            data=st.session_state.results_df.to_csv(index=False),
            file_name=output_filename,
            mime='text/csv'
        )

    # Create and display chart based on session state
    if st.session_state.show_chart:
        category_counts = st.session_state.results_df['topic'].value_counts()
        create_category_distribution_chart(category_counts)

    if st.session_state.show_chart2:
        create_embedding_projector(st.session_state.results_df)
