import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sentence_transformers import SentenceTransformer
import re

df_imdb = pd.read_csv("CSV/IMDBdata_MainData.csv")
df_imdb = df_imdb[['Title', 'Released', 'Genre', 'Director', 'Writer', 'Actors', 'Plot', 'Language', 'Country', 'Production']]

# Filter
def filtering(column: str):
    df_imdb[column] = df_imdb[column].astype(str)
    for i in range(len(df_imdb[column])):
        str_tmp = df_imdb[column][i]
        str_tmp = str_tmp.replace(",", "")
        df_imdb[column][i] = str_tmp

filtering('Genre')
filtering('Actors')
filtering('Director')
filtering('Country')

# Cleanup Writer
regex = r"\(([^\)]+)\)"
df_imdb['Writer'] = df_imdb['Writer'].astype(str)

for i in range(len(df_imdb['Writer'])):
    writer = df_imdb['Writer'][i]
    matches = re.finditer(regex, writer, re.MULTILINE)
    for matchNum, match in enumerate(matches, start=1):
        writer = writer.replace(match.group(), "")
    writer = writer.replace(",", "")
    df_imdb['Writer'][i] = writer

df_imdb.dropna(inplace=True)
df_imdb.to_csv('CSV/filter_imdb.csv')
df = pd.read_csv('CSV/filter_imdb.csv')
df.drop(columns=df.columns[0], axis=1, inplace=True)

def combine_features(row):
    return row['Title'] + ' ' + row['Released'] + ' ' + row['Genre'] + ' ' + row['Director'] + ' ' + row['Writer'] + ' ' + \
           row['Actors'] + ' ' + row['Plot'] + ' ' + row['Language'] + ' ' + row['Country'] + ' ' + row['Production']

df = df.astype(str)
df['combined_value'] = df.apply(combine_features, axis=1)
df['index'] = [i for i in range(0, len(df))]

df.to_csv("CSV/filter_imdb_bow.csv")

# Now, let's use TF-IDF for the movie recommender

df = pd.read_csv('CSV/filter_imdb_bow.csv')
df.drop(columns=df.columns[0], axis=1, inplace=True)

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_value'])

cosine_similarity_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(df['Title'])

def recommend(title, cosine_sim=cosine_similarity_matrix):
    recommended_movies = []
    if indices[indices == title].empty:
        return None
    idx = indices[indices == title].index[0]
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending=False)
    top_5_indices = list(score_series.iloc[1:6].index)
    for i in top_5_indices:
        recommended_movies.append(list(df['Title'])[i])
    return recommended_movies

def reply(query):
    top5 = recommend(query)
    if top5 is not None:
        top5_str = ""
        for i in top5:
            top5_str += i + ","
        template = f"""
          The given prompt may be answered using the top 5 list 
          {top5_str}
          """
    else:
        template = f"""
          But it seems that the prompt isn't recognizable by the database.
          So you suggest a movie title close to the prompt
          """
    
    return template

st.title("TF-IDF Movie Recommender")

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
        st.markdown(f"TF-IDF: {responses}")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": f"TF-IDF: {responses}"
        }
    )
