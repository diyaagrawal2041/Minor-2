#Used Google Collab for implemnetation of the project

import pandas as pd
import numpy as np
import seaborn as sns
import re
import matplotlib.pyplot as plt
import plotly
import plotly.express as px
from matplotlib import rcParams
from collections import Counter
from nltk.corpus import stopwords
import string

#importing dataset
from google.colab import files
uploaded= files.upload()

#reading csv file/dataset
Election_frame=pd.read_csv('us_election_2020.csv')
Election_frame.head(10)

#Size of the file
Election_frame.shape

#bar char and pie chart
rcParams["figure.figsize"]=20,20
fig,ax=plt.subplots(nrows=1,ncols=2)
sns.countplot(x="Subject",hue="Subject",data=Election_frame,ax=ax[0])
Election_frame.groupby('Subject').size().plot(kind='pie',ax=ax[1])

#line chart
rcParams["figure.figsize"]=10,10
X=Election_frame.groupby('Subject')['Subject'].transform('size')
sns.boxplot(x='Subject',y=X,data=Election_frame,hue='Subject',color='red',linewidth=3.5)

#count total number of users who tweeted
user_count=Election_frame.groupby('user')['user'].value_counts()
print("The no of user who tweeted", len(user_count))

#the users who have tweeted more than once
user_duplicates=Election_frame[Election_frame.duplicated()]
user_duplicates.count()

user_duplicates.groupby('user')['user'].size().plot(x=user_duplicates['user'],kind='pie',figsize=(23,25),legend=True,title='Duplicates')

#displaying 5 tweets of each candidate
Trump_tweets=Election_frame[Election_frame['Subject']=='Donald Trump']
Biden_tweets=Election_frame[Election_frame['Subject']=='Joe Biden']
Trump_tweets.head()
Biden_tweets.head()

#data cleaning
def clean_data(text):
  text=str(text).lower()
  text=re.sub('\[.*?\]','',text)
  text=re.sub('https?://\S+|www\.\S+','',text)
  text=re.sub('<.*?>+','',text)
  text=re.sub('[%s]' % re.escape(string.punctuation),'',text)
  text=re.sub('\n','',text)
  text=re.sub('\w*\d\w*','',text)
  return text
Trump_tweets['final_text']=Trump_tweets['text'].apply(lambda x:clean_data(x))
Biden_tweets['final_text']=Biden_tweets['text'].apply(lambda x:clean_data(x))

#remove stopwords
def remove_stopwords(x):
  return [w for w in x if not w in stop]
print("Text after removal of stopwords")
Trump_tweets.head()
print("Text after removal of stopwords")
Biden_tweets.head()

#finding positive, negative and neutral words in DOnald Trump Tweets
from textblob import TextBlob
from textblob import Word
positive_word=[]
negative_word=[]
neutral_word=[]
for word in Trump_tweets['final_text']:
  tokens=word.split()
  for Wr in tokens:
    stat=TextBlob(Wr)
    pol_stat=stat.polarity
    if(pol_stat>0.0):
      positive_word.append(Wr)
    elif(pol_stat<0.0):
      negative_word.append(Wr)
    else:
      neutral_word.append(Wr)

print("The no. of positive words in Donald Trump Tweets", len(positive_word))
print("The no. of negative words in Donald Trump Tweets", len(negative_word))
print("The no. of neutral words in Donald Trump Tweets", len(neutral_word))
#finding positive, negative and neutral words in Joe Biden Tweets
from textblob import TextBlob
from textblob import Word
positive_word=[]
negative_word=[]
neutral_word=[]
for word in Biden_tweets['final_text']:
  tokens=word.split()
  for Wr in tokens:
    stat=TextBlob(Wr)
    pol_stat=stat.polarity
    if(pol_stat>0.0):
      positive_word.append(Wr)
    elif(pol_stat<0.0):
      negative_word.append(Wr)
    else:
      neutral_word.append(Wr)

print("The no. of positive words in Joe Biden Tweets", len(positive_word))
print("The no. of negative words in Joe Biden Tweets", len(negative_word))
print("The no. of neutral words in Joe Biden Tweets", len(neutral_word))

#Sentimental Analysis
def cleaning_data(text):
    text=re.sub(r'@[A-Za-z0-9]+','',text)    # remove mentions
    text=re.sub(r'#','',text)                # remove # symbol
    text=re.sub(r'RT[\s]+','',text)          # remove RT
    text=re.sub(r'https?:\/\/\s+','',text)   # remove hyperlinks
    return text

Election_frame['clean_data']=Election_frame['text'].apply(cleaning_data)

def Sentimental_Analysis_1(df):
  text=df['clean_data']
  sentiment=[]
  from textblob import TextBlob
  overallsentiment=''
  for word in text:
    polar=TextBlob(word)
    polar_status=polar.sentiment.polarity
    sentiment.append(polar_status)
  df['Textblob_sentiment']=sentiment
  return(df)

frame=Sentimental_Analysis_1(Election_frame)
frame.info()

def sentiment(text):
  if(text>0.0):
    return('Positive')
  elif(text<0.0):
    return('Negative')
  else:
    return('Neutral')

frame['Sentiment']=frame['Textblob_sentiment'].apply(sentiment)
frame['Sentiment'].value_counts()

Trump=frame[frame['Subject']=='Donald Trump']
Biden=frame[frame['Subject']=='Joe Biden']

Trump['Sentiment'].value_counts()
Biden['Sentiment'].value_counts()

#displaying overall sentiments of Trump and Biden tweets in the form of chart
fig=px.pie(values=Trump['Sentiment'].value_counts(),hover_name=['Neural','Positive','Negative'],title="Overall Sentiments of Trump Tweets", labels={0:'Neutral',1:'Positive',2:'Negative'},hole=0.5,color=['Neutral','Positive','Negative'])
fig.show()
fig=px.pie(values=Biden['Sentiment'].value_counts(),hover_name=['Neural','Positive','Negative'],title="Overall Sentiments of Biden Tweets", labels={0:'Neutral',1:'Positive',2:'Negative'},hole=0.5,color=['Neutral','Positive','Negative'])
fig.show()

#displaying overall sentiments of trump and Biden tweets in the form of violin plot
fig=px.violin(frame,x='Sentiment',y='Subject',color='Sentiment',hover_name='Sentiment',violinmode='overlay',box=True,orientation='v',title='Joe Biden V/S Donald Trump')
fig.show()
