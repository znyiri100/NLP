import streamlit as st
#from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
#from langchain.output_parsers import PydanticOutputParser
#from pydantic import BaseModel, Field
import json
#from pprint import pprint
from langchain_openai import ChatOpenAI
#import random
import pandas as pd
import time
from datetime import datetime

word_list_default = [
    "apple", "computer", "banana", "laptop", "orange",
    "smartphone", "grape", "tablet", "mango", "keyboard",
    "pear", "mouse", "strawberry", "printer", "blueberry","monitor", "raspberry", "server", "peach", "router", "house"
]

def gen_topics(word_list=word_list_default, num_topics=3, prompt_tail=''):
#def gen_topics(word_list=["apple", "computer", "banana", "laptop", "orange"], num_topics=3, prompt_tail=''):
    ######################
    # Identify 3 topics each word may belong to, consolidate those into given number of main topics
    # prompt_tail is to append extra info at the end of the prompt generating the main topics, e.g. to avoid certain topic names
    ######################

    system_message = SystemMessage(
        content="You are an expert at categorizing words. For each word, list 2-3 potential topic categories it could belong to."
    )

    human_message = HumanMessagePromptTemplate.from_template(
        """Analyze this word and suggest 2-3 broad topics it could belong to: {word}

        Return your response in this exact JSON format:
        {{
            "word": "{word}",
            "topics": ["topic1", "topic2", "topic3"]
        }}"""
    )

    prompt = ChatPromptTemplate.from_messages([system_message, human_message])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    #chain = LLMChain(llm=llm, prompt=prompt)
    chain = prompt | llm
    #print(prompt.format(word=words))

    # words = [{"word": word} for word in word_list]
    # word_analyses = chain.batch(words)

    # Initialize progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Prepare batch inputs
    inputs = [{"word": word} for word in word_list]
    
    # Process in batches
    batch_size = max(len(word_list) // 5, 10)  # Ensure minimum batch size of 10
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
    word_analyses=results

    #print( 'Out of ', len(word_list), 'words we categorized ', len(word_analyses))
    #st.info(word_analyses)  # words with the assigned 3 broad topics

    ######################
    # consolidate broad categoris into X main categories
    ######################

    # Define the prompt template
    system_message = SystemMessage(    content="You are an expert at categorizing topics into main topics.")
    human_message = HumanMessagePromptTemplate.from_template(
    """Given these topics:
    {analyses_str}

    Create exactly {num_topics} main topics.
    Return your response in this exact JSON format:
    {{
        "topics": ["topic1", "topic2", "topic3"]
    }}
    {prompt_tail}
    """
    )
    consolidate_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

    # Prepare the analyses string
    analyses_str = set()
    for analysis in word_analyses:
        #print(analysis.content)
        data = json.loads(analysis.content)
        analyses_str.update(data['topics'])

    # consolidate topics
    st.info( f"Consolidating {len(list(analyses_str))} categories into {num_topics} main topics:\n\n{list(analyses_str)}")
    #consolidate_chain = LLMChain(llm=llm, prompt=consolidate_prompt)
    consolidate_chain = consolidate_prompt | llm
    response = consolidate_chain.invoke({"analyses_str": analyses_str, "num_topics": num_topics, "prompt_tail": prompt_tail})

    # handling mailformed json
    try:
        topics=json.loads(response.content)['topics']
    except:
        topics=json.loads(response.content.replace('```json', '').replace('```', ''))['topics']

    #print('Consolidated ', len(list(analyses_str)), ' broad categories into ', len(topics), ' main topics')
    return topics

def app():

    st.title("Discover topics for list of words")

    # Input method selection
    input_method = st.sidebar.radio("Select input method:", ("Word List", "File"))

    word_list_df=pd.DataFrame()
    word_list_default = [
        "apple", "computer", "banana", "laptop", "orange", "smartphone",
        "grape", "tablet", "mango", "keyboard", "pear", "mouse",
        "strawberry", "printer", "blueberry", "monitor", "raspberry",
        "server", "peach", "router", "house"
    ]

    if input_method == "Word List":
        word_list_df=pd.DataFrame({'word': word_list_default})
        st.session_state.results_df = None
        # if word_list_input:
        #     word_list = [word.strip() for word in word_list_input.replace(',', '\n').splitlines() if word.strip()]
    else:
        uploaded_file = st.sidebar.file_uploader("Upload file with one word per line:", type=["txt"])
        if uploaded_file is not None:
            try:
                word_list_df = pd.read_csv(uploaded_file, names=['word'])
            except Exception as e:
                st.error(f"Error reading the file: {e}")
        st.session_state.results_df = None
        
    if word_list_df.shape[0]>0:

        # Add word selection options in sidebar
        #st.sidebar.subheader("Word Selection")
        selection_method = st.sidebar.radio(
            "Select words by:",
            ["First N words", "Random sample"]
        )
        
        # Add input box for number of words to process
        n_words = st.sidebar.number_input(
            f"Words (max {word_list_df.shape[0]})",
            min_value=1,
            max_value=word_list_df.shape[0],
            value=min(10, word_list_df.shape[0]),
            step=1
        )
        
        # Process words based on selection method
        if selection_method == "Random sample":
            #word_list_df_selected = word_list_df.sample(n=n_words, random_state=42)
            word_list_df_selected = word_list_df.sample(n=n_words)
        else:
            word_list_df_selected = word_list_df.head(n_words)

        word_list_input = st.text_area("Enter words (comma-separated or one per line):", value='\n'.join(word_list_df_selected['word'].tolist()))
        if word_list_input:
            word_list = [word.strip() for word in word_list_input.replace(',', '\n').splitlines() if word.strip()]

        word_list_df_selected = pd.DataFrame({'word': word_list_input.split('\n')})
        st.write(f"{word_list_df_selected.shape[0]} words")

        num_topics = st.sidebar.number_input("Number of Topics:", min_value=2, value=3)
        prompt_tail = st.sidebar.text_input("Enter additional instructions for topic generation (e.g., 'Do not generate topics with food')", value='')

        # Run the selected clustering method
        if st.sidebar.button("Go for it!"):
            topic_list = gen_topics( word_list_df_selected['word'].tolist(), num_topics=num_topics, prompt_tail=prompt_tail )
            topic_list_df = pd.DataFrame(topic_list, columns=['topics'])
            st.write(topic_list_df)