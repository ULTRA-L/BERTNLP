import pandas as pd
import re
from sentence_transformers import SentenceTransformer

df_imdb = pd.read_csv("CSV/IMDBdata_MainData.csv")
df_imdb = df_imdb[['Title','Released','Genre','Director','Writer','Actors','Plot','Language','Country','Production']]

#Filter
def filtering(column:str):
  df_imdb[column]=df_imdb[column].astype(str)
  for i in range(len(df_imdb[column])):
    str_tmp = df_imdb[column][i]
    str_tmp = str_tmp.replace(",","")
    df_imdb[column][i] = str_tmp

filtering('Genre')
filtering('Actors')
filtering('Director')
filtering('Country')

#Cleanup Writer
regex = r"\(([^\)]+)\)"
df_imdb['Writer']=df_imdb['Writer'].astype(str)

for i in range(len(df_imdb['Writer'])):
  writer = df_imdb['Writer'][i]
  matches = re.finditer(regex,writer,re.MULTILINE)
  for matchNum, match in enumerate(matches, start=1):
    writer = writer.replace(match.group(),"")
  writer = writer.replace(",", "")
  df_imdb['Writer'][i] = writer

df_imdb.dropna(inplace=True)
df_imdb.to_csv('CSV/filter_imdb.csv')
df = pd.read_csv('CSV/filter_imdb.csv')
df.drop(columns = df.columns[0], axis = 1, inplace=True)

def combine_features(row):
    return row['Title']+' '+row['Released']+' '+ row['Genre']
    +' '+row['Director']+' '+row['Writer']+' '+row['Actors']+' '+row['Plot']
    +' '+row['Language']+' '+row['Country']+' '+row['Production']

df = df.astype(str)
df['combined_value'] = df.apply(combine_features,axis=1)
df['index'] = [i for i in range(0,len(df))]

bert = SentenceTransformer('bert-base-nli-mean-tokens')
sentence_embeddings = bert.encode(df['combined_value'].tolist())

df_bert = pd.DataFrame(sentence_embeddings)
df_bert.to_csv("CSV/bert.csv")

print("bert.csv was saved")