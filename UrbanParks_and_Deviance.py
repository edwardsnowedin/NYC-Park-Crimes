#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()


# ## Cleaning Data

# ## 2018

# In[54]:


pc_2018_1QTR = pd.read_excel('Class_Materials/Data/pc_q1_2018.xlsx', skiprows = [0,1,2])
pc_2018_1QTR = pc_2018_1QTR.rename(columns = {'Murder': 'MURDER'})
pc_2018_1QTR['Quarterly'] = 0 #Q1
pc_2018_1QTR.head()


# In[55]:


pc_2018_1QTR = pc_2018_1QTR.drop(1154)
pc_2018_1QTR.tail()


# In[56]:


pc_2018_1QTR.isnull().sum()


# In[57]:


pc_2018_1QTR = pc_2018_1QTR.dropna()


# In[58]:


pc_2018_2QTR = pd.read_excel('Class_Materials/Data/pc_q2_2018.xlsx', skiprows = [0,1,2,3])
pc_2018_2QTR['Quarterly'] = 1 #Q2
pc_2018_2QTR.head()


# In[59]:


pc_2018_2QTR = pc_2018_2QTR.drop(1154)
pc_2018_2QTR.tail()


# In[60]:


pc_2018_2QTR.isnull().sum()


# In[61]:


pc_2018_2QTR = pc_2018_2QTR.dropna()


# In[62]:


pc_2018_3QTR = pd.read_excel('Class_Materials/Data/pc_q3_2018.xlsx', skiprows = [0,1,2])
pc_2018_3QTR['Quarterly'] = 2 #Q3
pc_2018_3QTR.head()


# In[63]:


pc_2018_3QTR = pc_2018_3QTR.drop(1154)

pc_2018_3QTR.tail()


# In[64]:


pc_2018_3QTR.isnull().sum()


# In[65]:


pc_2018_3QTR = pc_2018_3QTR.dropna()


# In[66]:


pc_2018_4QTR = pd.read_excel('Class_Materials/Data/pc_q4_2018.xlsx', skiprows = [0,1,2])
pc_2018_4QTR['Quarterly'] = 3 #Q4
pc_2018_4QTR.head()


# In[67]:


pc_2018_4QTR = pc_2018_4QTR.drop(1154)
pc_2018_4QTR.tail()


# In[68]:


pc_2018_4QTR.isnull().sum()


# In[69]:


pc_2018_4QTR = pc_2018_4QTR.dropna()


# In[70]:


df2018 = pc_2018_1QTR.append([pc_2018_2QTR, pc_2018_3QTR  ,pc_2018_4QTR], sort = False)


# In[71]:


df2018["Year"] = 2018


# In[72]:


df2018.head()


# In[73]:


df2018.tail()


# ## 2017

# In[74]:


pc_2017_1QTR = pd.read_excel('Class_Materials/Data/pc_q1_2017.xlsx', skiprows = [0,1,2])
pc_2017_1QTR['Quarterly'] = 0 #Q1
pc_2017_1QTR.head()


# In[75]:


pc_2017_1QTR = pc_2017_1QTR.drop(1154)
pc_2017_1QTR.tail()


# In[76]:


pc_2017_1QTR.isnull().sum()


# In[77]:


pc_2017_1QTR = pc_2017_1QTR.dropna()


# In[78]:


pc_2017_2QTR = pd.read_excel('Class_Materials/Data/pc_q2_2017.xlsx', skiprows = [0,1,2])
pc_2017_2QTR['Quarterly'] = 1 #Q2
pc_2017_2QTR.head()


# In[79]:


pc_2017_2QTR = pc_2017_2QTR.drop(1154)
pc_2017_2QTR.tail()


# In[80]:


pc_2017_2QTR.isnull().sum()


# In[81]:


pc_2017_2QTR = pc_2017_2QTR.dropna()


# In[82]:


pc_2017_3QTR = pd.read_excel('Class_Materials/Data/pc_q3_2017.xlsx', skiprows = [0,1,2])
pc_2017_3QTR['Quarterly'] = 2 #'Q3'
pc_2017_3QTR.head()


# In[83]:


pc_2017_3QTR = pc_2017_3QTR.drop(1154)
pc_2017_3QTR.tail()


# In[84]:


pc_2017_3QTR.isnull().sum()


# In[85]:


pc_2017_3QTR = pc_2017_3QTR.dropna()


# In[86]:


pc_2017_4QTR = pd.read_excel('Class_Materials/Data/pc_q4_2017.xlsx', skiprows = [0,1,2])
pc_2017_4QTR = pc_2017_4QTR.rename(columns = {'Murder': 'MURDER'})
pc_2017_4QTR['Quarterly'] = 3 #'Q4'
pc_2017_4QTR.head()


# In[87]:


pc_2017_4QTR = pc_2017_4QTR.drop(1154)
pc_2017_4QTR.tail()


# In[88]:


pc_2017_4QTR.isnull().sum()


# In[89]:


pc_2017_4QTR = pc_2017_4QTR.dropna()


# In[90]:


df2017 = pc_2017_1QTR.append([pc_2017_2QTR, pc_2017_3QTR  ,pc_2017_4QTR], sort = False)
df2017['Year'] = 2017
df2017.head()


# In[91]:


df2017.tail()


# ## 2016

# In[92]:


pc_2016_1QTR = pd.read_excel('Class_Materials/Data/pc_q1_2016.xlsx', skiprows = [0,1,2])
pc_2016_1QTR['Quarterly'] = 0 #'Q1'
pc_2016_1QTR.head()


# In[93]:


pc_2016_1QTR = pc_2016_1QTR.drop([1154,1155])
pc_2016_1QTR.tail()


# In[94]:


pc_2016_1QTR.isnull().sum()


# In[95]:


pc_2016_1QTR = pc_2016_1QTR.dropna()


# In[96]:


pc_2016_2QTR = pd.read_excel('Class_Materials/Data/pc_q2_2016 .xlsx', skiprows = [0,1,2])
pc_2016_2QTR['Quarterly'] = 1 #'Q2'
pc_2016_2QTR.head()


# In[97]:


pc_2016_2QTR = pc_2016_2QTR.drop(1154)
pc_2016_2QTR.tail()


# In[98]:


pc_2016_2QTR.isnull().sum()


# In[99]:


pc_2016_2QTR = pc_2016_2QTR.dropna()


# In[100]:


pc_2016_3QTR = pd.read_excel('Class_Materials/Data/pc_q3_2016 .xlsx', skiprows = [0,1,2])
pc_2016_3QTR['Quarterly'] = 2 #'Q3'
pc_2016_3QTR.head()


# In[101]:


pc_2016_3QTR = pc_2016_3QTR.drop(1154)
pc_2016_3QTR.tail()


# In[102]:


pc_2016_3QTR.isnull().sum()


# In[103]:


pc_2016_3QTR = pc_2016_3QTR.dropna()


# In[104]:


pc_2016_4QTR = pd.read_excel('Class_Materials/Data/pc_q4_2016 .xlsx', skiprows = [0,1,2])
pc_2016_4QTR['Quarterly'] = 3 #'Q4'
pc_2016_4QTR.head()


# In[105]:


pc_2016_4QTR = pc_2016_4QTR.drop(1154)
pc_2016_4QTR.tail()


# In[106]:


pc_2016_4QTR.isnull().sum()


# In[107]:


pc_2016_4QTR = pc_2016_4QTR.dropna()


# In[108]:


df2016 = pc_2016_1QTR.append([pc_2016_2QTR, pc_2016_3QTR, pc_2016_4QTR], sort = False)
df2016['Year'] = 2016 
df2016.head()


# In[109]:


df2016.tail()


# In[110]:


pc_2015_1QTR = pd.read_excel('Class_Materials/Data/pc_q1_2015.xlsx', skiprows = [0,1,2,3])
pc_2015_1QTR['Quarterly'] = 0 #'Q1'
pc_2015_1QTR.head()


# In[111]:


pc_2015_1QTR = pc_2015_1QTR.drop(1154)
pc_2015_1QTR.tail()


# In[112]:


pc_2015_1QTR.isnull().sum()


# In[113]:


pc_2015_1QTR = pc_2015_1QTR.dropna()


# In[114]:


pc_2015_2QTR = pd.read_excel('Class_Materials/Data/pc_q2_2015.xlsx', skiprows = [0,1,2,3])
pc_2015_2QTR['Quarterly'] = 1 #'Q2'
pc_2015_2QTR.head()


# In[115]:


pc_2015_2QTR = pc_2015_2QTR.drop(1154)
pc_2015_2QTR.tail()


# In[116]:


pc_2015_2QTR.isnull().sum()


# In[117]:


pc_2015_2QTR = pc_2015_2QTR.dropna()


# In[118]:


pc_2015_3QTR = pd.read_excel('Class_Materials/Data/pc_q3_2015.xlsx', skiprows = [0,1,2,3])
pc_2015_3QTR['Quarterly'] = 2 #'Q3'
pc_2015_3QTR.head()


# In[119]:


pc_2015_3QTR = pc_2015_3QTR.drop(1154)
pc_2015_3QTR.tail()


# In[120]:


pc_2015_3QTR.isnull().sum()


# In[121]:


pc_2015_3QTR = pc_2015_3QTR.dropna()


# In[122]:


pc_2015_4QTR = pd.read_excel('Class_Materials/Data/pc_q4_2015.xlsx', skiprows = [0,1,2])
pc_2015_4QTR['Quarterly'] = 3 #'Q4'
pc_2015_4QTR.head()


# In[123]:


pc_2015_4QTR = pc_2015_4QTR.drop(1154)
pc_2015_4QTR.tail()


# In[124]:


pc_2015_4QTR.isnull().sum()


# In[125]:


pc_2015_4QTR = pc_2015_4QTR.dropna()


# In[126]:


df2015 = pc_2015_1QTR.append([pc_2015_2QTR,pc_2015_3QTR,pc_2015_4QTR], sort = False)
df2015['Year'] = 2015
df2015.head()


# In[127]:


df2015.tail()


# In[128]:


nycParkCrimes_df = df2015.append([df2016,df2017,df2018], sort = False)
nycParkCrimes_df = nycParkCrimes_df.rename(columns = {'SIZE (ACRES)': 'ACRES'})
nycParkCrimes_df = nycParkCrimes_df.rename(columns = {'GRAND LARCENY OF MOTOR VEHICLE': 'GRAND_LARCENY_OF_MOTOR_VEHICLE'})
nycParkCrimes_df = nycParkCrimes_df.rename(columns = {'GRAND LARCENY': 'GRAND_LARCENY'})
nycParkCrimes_df = nycParkCrimes_df.rename(columns = {'FELONY ASSAULT': 'FELONY_ASSAULT'})
nycParkCrimes_df.head()


# In[129]:


nycParkCrimes_df.tail()


# In[130]:


nycParkCrimes_df = nycParkCrimes_df[nycParkCrimes_df.BOROUGH != 'BROOKLYN/QUEENS']
nycParkCrimes_df['BOROUGH'].unique()


# In[131]:


nycParkCrimes_df['CATEGORY'].unique()


# ## Descriptive Statistics

# In[132]:


nycParkCrimes_df['ACRES'].corr(nycParkCrimes_df['TOTAL'])


# In[133]:


#nycParkCrimes_df['SIZE (ACRES)'] = pd.to_numeric(nycParkCrimes_df['SIZE (ACRES)'])
nycParkCrimes_df['MURDER'] = pd.to_numeric(nycParkCrimes_df['MURDER'])
nycParkCrimes_df['RAPE'] = pd.to_numeric(nycParkCrimes_df['RAPE'])
nycParkCrimes_df['ROBBERY'] = pd.to_numeric(nycParkCrimes_df['ROBBERY'])
nycParkCrimes_df['FELONY ASSAULT'] = pd.to_numeric(nycParkCrimes_df['FELONY_ASSAULT'])
nycParkCrimes_df['BURGLARY'] = pd.to_numeric(nycParkCrimes_df['BURGLARY'])
nycParkCrimes_df['GRAND LARCENY'] = pd.to_numeric(nycParkCrimes_df['GRAND_LARCENY'])
nycParkCrimes_df['GRAND LARCENY OF MOTOR VEHICLE'] = pd.to_numeric(nycParkCrimes_df['GRAND_LARCENY_OF_MOTOR_VEHICLE'])
nycParkCrimes_df['TOTAL'] = pd.to_numeric(nycParkCrimes_df['TOTAL'])


# In[134]:


nycParkCrimes_df = pd.DataFrame(nycParkCrimes_df)
nycParkCrimes_df.dtypes


# In[135]:


annual_crime = nycParkCrimes_df.groupby(['Year'], as_index = False).agg({'MURDER': np.sum, 'RAPE': np.sum, 
                                                                         'ROBBERY': np.sum,'FELONY_ASSAULT': np.sum, 
                                                                         'BURGLARY': np.sum,'GRAND_LARCENY': np.sum,
                                                                         'GRAND_LARCENY_OF_MOTOR_VEHICLE': np.sum, 'TOTAL': np.sum})

annual_crime


# In[136]:


ax = annual_crime.plot(x='Year', y=['MURDER', 'ROBBERY','FELONY_ASSAULT','BURGLARY', 'GRAND_LARCENY', 'GRAND_LARCENY_OF_MOTOR_VEHICLE'], legend = False)
ax.locator_params(integer=True)
plt.show()


# In[137]:


total_crime = annual_crime.plot(x='Year', y='TOTAL')
total_crime.locator_params(integer=True)


# In[138]:


nycParkCrimes_df.groupby("Year").TOTAL.sum()


# In[139]:


nycParkCrimes_df.groupby("PARK").TOTAL.sum().sort_values(ascending=False)[:10]


# In[140]:


nycParkCrimes_df.groupby("PARK").TOTAL.sum().sort_values(ascending=False)[:10].plot.bar()


# In[141]:


nycParkCrimes_df.groupby("BOROUGH").TOTAL.sum()


# In[142]:


nycParkCrimes_df.groupby("BOROUGH").TOTAL.sum().sort_values(ascending=False).plot.bar()


# In[143]:


nycParkCrimes_df.groupby("CATEGORY").TOTAL.sum().sort_values(ascending=False)[:6].plot.bar()


# In[145]:


nycParkCrimes_df.groupby("Quarterly").TOTAL.sum()


# In[146]:


nycParkCrimes_df.groupby("Quarterly").TOTAL.sum().sort_values(ascending=False).plot.bar()


# In[147]:


df_pc = pd.DataFrame(nycParkCrimes_df,columns=['ACRES','CATEGORY',
                                                 'Quarterly','MURDER',
                                                 'FELONY_ASSAULT','BURGLARY',
                                                 'GRAND_LARCENY', 'GRAND_LARCENY_OF_MOTOR_VEHICLE', 
                                                 'TOTAL'])




corr = df_pc.corr()

plt.figure(figsize=(10,8))

ax = sns.heatmap(corr, 
                 vmin=-1, 
                 vmax=1,
                 cmap='bwr',
                 annot=True,
                )

bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)


# ## Predictive modeling - Decision Tree Classifier

# In[148]:


from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation


# In[149]:


X = nycParkCrimes_df[['ACRES', 'Quarterly']] # Features
y = nycParkCrimes_df['TOTAL'] # Target variable


# In[150]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0) 


# In[151]:


# Create Decision Tree classifer object
#Without optimization
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)


# In[152]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred) * 100) 


# In[153]:


from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

dot_data = StringIO()
export_graphviz(clf, 
                out_file=dot_data,  
                filled=True, 
                rounded=True,
                special_characters=True,feature_names = X.columns)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# ## Once the model was complete, I evaluated how accurately the classifier predicted the target variable.  An accuracy score of 87.6 percent was the result, which was relatively high. When it came to the visualization of the Decision Tree, the original model produced an unpruned tree, and as a result, the tree was difficult to comprehend. Consequently, I pruned the Decision Tree for optimal performance.

# In[154]:


# Create Decision Tree classifer object
#With Optimization
clf_deux = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train Decision Tree Classifer
clf_deux = clf_deux.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf_deux.predict(X_test)


# In[155]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred) * 100) 


# In[156]:


from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

dot_data = StringIO()
export_graphviz(clf_deux, 
                out_file=dot_data,  
                filled=True, 
                rounded=True,
                special_characters=True,feature_names = X.columns)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# ## Once pruned, the model produced an accuracy score of 88.64, which was a modest increase from the previous score. Furthermore, the pruned visualization of the tree is much more comprehensible.

# ## Topic Modeling

# In[3]:


pip install --user GetOldTweets3


# In[52]:


consumer_key= '#'
consumer_secret= '#'
access_token= '#'
access_token_secret= '#'


# In[16]:


import tweepy as tw
import csv


# In[17]:


auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)


# In[51]:


#search_words = "#flushingmeadowscorona"


# In[46]:


#for tweet in tw.Cursor(api.search,q=search_words,count=100,\
                           #lang="en",\
                           #since_id='2019-12-19').items():
    #print(tweet.created_at, tweet.text)


# In[6]:


import GetOldTweets3 as got

tweetCriteria = got.manager.TweetCriteria().setQuerySearch('morningsidepark')                                           .setNear("New York")                                           .setSince("2019-12-19")                                           .setUntil("2020-01-19")                                           .setMaxTweets(50)
            

tweet_df = pd.DataFrame({'got_criteria':got.manager.TweetManager.getTweets(tweetCriteria)})


# In[7]:


#Source: https://medium.com/@robbiegeoghegan/download-twitter-data-with-10-lines-of-code-42eb2ba1ab0f
def get_twitter_info():
    tweet_df["tweet_text"] = tweet_df["got_criteria"].apply(lambda x: x.text)
    tweet_df["date"] = tweet_df["got_criteria"].apply(lambda x: x.date)
    tweet_df["hashtags"] = tweet_df["got_criteria"].apply(lambda x: x.hashtags)
    tweet_df["link"] = tweet_df["got_criteria"].apply(lambda x: x.permalink)


# In[8]:


tweet_df = pd.DataFrame({'got_criteria':got.manager.TweetManager.getTweets(tweetCriteria)})
get_twitter_info()


# In[9]:


tweet_df.tail()


# In[10]:


tweet_df.shape


# In[11]:


get_ipython().system('pip3.6 install --user langdetect')


# In[12]:


from langdetect import detect


# In[13]:


tweet_df['lang'] = tweet_df.tweet_text.apply(detect)


# In[14]:


tweet_df.head()


# In[15]:


tweet_df.lang.value_counts()


# In[16]:


tweet_df = tweet_df.loc[tweet_df.lang=='en']


# In[17]:


import gensim
from gensim.matutils import Sparse2Corpus
from gensim import corpora, models, similarities
from gensim.corpora import Dictionary, MmCorpus
from nltk.corpus import stopwords
import nltk 
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import re

import pandas as pd
import numpy as np
import time
from dateutil.parser import parse
import requests
import string
from collections import Counter

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[18]:


def token_process(doc):
    
    ## stop words and updates
    ## Note, you should add more terms to this list to see what may or may not be useful.
    ## Also note, that I also remove punctuation here by adding the string module
    stop_en = stopwords.words('english') + list(string.punctuation) + [u'...',u',',u'.',u'?',u'!',u':',u';', u')', u'(',u'[',u']',u'{',u'}',u'%',u'@',u'-',u'`',
                                           u'san',u'francisco',u'san francisco',u'new',u'tr',u'th',u'to',u'on',u'of',u'mr',
                                           u'monday','tuesday',u'wednesday',u'thursday',u'friday',u'saturday',u'sunday','want','befor','becaus'
                                           u'said',u'ms',u'york',u'say',u'could',u'q',u'got',u'found',u'began','|',"''","'s","``","--",
                                           'mr','year','would','one','way','l','ms.','$','mr.','dr.','get','before','like','know','day','because',
                                           '"','see','look','dont','im','&','b','also','de','la','el','en','un','two','al','su','es','lo','se']
    
        
    #stemming
    stemmer = SnowballStemmer("english")
    
    #lemmatizer
    lemmatizer = WordNetLemmatizer() 
    
    #tokenize
    tokens = [w.strip() for sent in sent_tokenize(doc) for w in word_tokenize(sent)] if doc else None
    
    #remove numbers
    num_pat = re.compile(r'^(-|\+)?(\d*).?(\d+)')
    tokens = filter(lambda x: not num_pat.match(x), tokens)
    
    #remove dates
    date_pat =  re.compile(r'^(\d{1,2})(/|-)(\d{1,2})(/|-)(\d{2,4})$')
    tokens = filter(lambda x: not date_pat.match(x), tokens)
    
    #use stemmer
    stemmed_tokens = map(lambda x: stemmer.stem(x), tokens)
    
    #filter out empty tokens and stopwords
    stemmed_tokens = filter(lambda x: x and x.strip() not in stop_en, stemmed_tokens)

    #use lemmatizer
    lemmatized_and_stemmed_tokens = map(lambda x: lemmatizer.lemmatize(x), stemmed_tokens)

    #again filter out empty tokens and stopwords
    lemmatized_and_stemmed_tokens = filter(lambda x: x and x.strip() not in stop_en, lemmatized_and_stemmed_tokens)

    #remove any lingering white space tokens
    lemmatized_and_stemmed_tokens = filter(lambda x: x and x.strip() not in [u' '],lemmatized_and_stemmed_tokens)

    x = ' '.join(lemmatized_and_stemmed_tokens)
    return x.split(' ')


# In[19]:


tweet_df["clean_tweet"] = tweet_df["tweet_text"].apply(lambda doc: token_process(doc))


# In[20]:


tweet_df.clean_tweet.head()


# In[21]:


initial_corpus = tweet_df["clean_tweet"].tolist()


# In[22]:


dictionary_LDA = corpora.Dictionary(initial_corpus)


# In[23]:


dictionary_LDA.filter_extremes(no_below=3)


# In[24]:


corpus = [dictionary_LDA.doc2bow(doc_) for doc_ in initial_corpus]


# In[25]:


corpus[0]


# In[26]:


dictionary_LDA[2]


# In[27]:


num_topics = 20 # Number of Topics, we set initially as K. 


# In[28]:


np.random.seed(123456)


# In[29]:


lda_model = models.LdaMulticore(corpus,                            num_topics=num_topics,                                  id2word=dictionary_LDA,                                  alpha=[0.01]*num_topics)


# In[30]:


for i,topic in lda_model.show_topics(formatted=True, num_topics=num_topics, num_words=20):
    print(str(i)+": "+ topic)
    print()


# In[31]:


lda_model[corpus[0]]


# In[32]:


get_ipython().system('pip3.6 install --user pyLDAvis')
get_ipython().system('pip3.6 install --user joblib')


# In[33]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pyLDAvis
import pyLDAvis.gensim
from joblib import parallel_backend


# In[34]:


with parallel_backend('threading'):
    vis = pyLDAvis.gensim.prepare(topic_model=lda_model, corpus=corpus, dictionary=dictionary_LDA)


# In[50]:


pyLDAvis.enable_notebook()
pyLDAvis.display(vis)


# In[102]:


import re
import nltk


# In[103]:


def removal(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
        
    return input_txt


# In[104]:


tweet_df['clean_tweet'] = np.vectorize(removal)(tweet_df['tweet_text'], "@[\w]*")


# In[105]:


tweet_df['clean_tweet'] = tweet_df['clean_tweet'].str.replace("[^a-zA-Z#]", " ")


# In[106]:


tokenized_tweets = tweet_df['clean_tweet'].apply(lambda x: x.split())
tokenized_tweets.head()


# In[56]:


from nltk.stem.porter import *
stemmer = PorterStemmer()

tokenized_tweets = tokenized_tweets.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
tokenized_tweets.head()


# In[36]:


pip install --user wordcloud


# In[37]:


from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# In[ ]:





# In[38]:


words = ' '.join([text for text in tweet_df['tweet_text']])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


# In[39]:


from gensim.models import CoherenceModel


# In[40]:


# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model,                                     texts=initial_corpus,                                     dictionary=dictionary_LDA,                                     coherence='c_v')


# In[41]:


coherence_lda = coherence_model_lda.get_coherence()


# In[42]:


coherence_lda * 100


# In[43]:


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=5):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics_K in range(start, limit, step):
        lda_model_K = models.LdaMulticore(corpus,                            num_topics=num_topics_K,                                  id2word=dictionary,                                  alpha=[0.01]*num_topics_K)
        
        model_list.append(lda_model_K)
        coherencemodel = CoherenceModel(model=lda_model_K, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


# In[44]:


model_list, coherence_values = compute_coherence_values(dictionary=dictionary_LDA,                                                        corpus=corpus,                                                        texts=initial_corpus,                                                        start=5,                                                        limit=30,                                                        step=5)


# In[49]:


import matplotlib.pyplot as plt
limit=30; start=5; step=5;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.locator_params(integer=True)
plt.show()


# In[47]:


#https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
for m, cv in zip(x, coherence_values):
    print('Num Topics =', m, " has Coherence Value of", round(cv,4))


# In[ ]:




