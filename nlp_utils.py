import pandas as pd
import streamlit as st
import os

NLP_FOLDER_PATH='.'

word_list_default = [
    "apple", "computer", "banana", "laptop", "orange", "smartphone",
    "grape", "tablet", "mango", "keyboard", "pear", "mouse",
    "strawberry", "printer", "blueberry", "monitor", "raspberry",
    "server", "peach", "router", "house"
]

def f_word_list():

  # Read folder for files with name word_list*.txt
  word_list_files = [f for f in os.listdir(NLP_FOLDER_PATH) if f.startswith('word_list') ]
  word_list_files.sort()
  #st.write("Available word list files:", word_list_files)

  # Input method selection
  input_method = st.sidebar.radio("Select input method:", (["Default Word List"] + word_list_files + ["Upload File"]))
  
  if input_method == "Default Word List":
      word_list_df=pd.DataFrame({'word': word_list_default})
  elif input_method.startswith('word_list'):
      try:
          word_list_df = pd.read_csv(NLP_FOLDER_PATH+'/'+input_method, names=['word'])
          # word_list=list(word_list_df['word'])
      except Exception as e:
          st.error(f"Error reading the file: {e}")
  else:
      uploaded_file = st.sidebar.file_uploader("Upload file with one word per line:", type=["txt"])
      if uploaded_file is not None:
          try:
              word_list_df = pd.read_csv(uploaded_file, names=['word'])
          except Exception as e:
              st.error(f"Error reading the file: {e}")
      else:
          word_list_df=pd.DataFrame()

  return word_list_df