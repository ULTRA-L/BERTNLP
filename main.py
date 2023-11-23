import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

df = pd.read_csv('CSV/filter_imdb_bow.csv')
df.drop(columns = df.columns[0], axis=1, inplace = True)

sentence_embeddings = pd.read_csv('CSV/bert.csv')
sentence_embeddings.drop(columns = sentence_embeddings.columns[0], axis=1, inplace = True)

similarity = cosine_similarity(sentence_embeddings)
indices = pd.Series(df['Title'])

def recommend(title, cosine_sim = similarity):
  recommended_movies = []
  if indices[indices == title].empty:
    return None
  idx = indices[indices == title].index[0]
  score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)
  top_5_indices = list(score_series.iloc[1:6].index)
  for i in top_5_indices:
    recommended_movies.append(list(df['Title'])[i])
  return recommended_movies

client = OpenAI(
    api_key = st.secrets["bert_openai_key"]
)

def reply(query):
  top5 = recommend(query)
  if top5 is not None:
    top5_str = ""
    for i in top5:
      top5_str += i + ","

    template = f"""
      The given prompt may be answer using the top 5 list 
      {top5_str}
      """
  else:
    template = f"""
      But it seems that the prompt isn't recognizable by the database.
      So you suggest a movie title close to the prompt
      """

  completion = client.chat.completions.create(
    model='gpt-3.5-turbo-0301',
    messages=[
      {
        "role": "system",
        "content":
          """
            You are a Movie Recommender Chatbot named MovieBuddy.
            Your purpose is to recommend atleast top 5 Movies according to the user's response
            You usually reply in a friendly manner
          """ + template
      },
      {
        "role": "user",
        "content": query
      }
    ]
  )

  response = completion.choices[0].message.content
  return response

st.title("BERT Movie Recommender")

if "messages" not in st.session_state:
  st.session_state.messages = []

for message in st.session_state.messages:
  with st.chat_message(message["role"]):
    st.markdown(message["content"])

if prompt := st.chat_input("Enter a Movie Title"):
  st.chat_message("user").markdown(prompt)

  st.session_state.messages.append(
    {
      "role": "user",
      "content": prompt
    }
  )

  responses = reply(prompt)

  with st.chat_message("assistant"):
    #for response in responses:
    st.markdown(f"BERT: {responses}")

  #for response in responses:
  st.session_state.messages.append(
    {
      "role": "assistant",
      "content": f"BERT: {responses}"
    }
  )