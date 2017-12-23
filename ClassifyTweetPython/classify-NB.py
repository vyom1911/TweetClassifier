# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 22:28:00 2017

@author: Vyom
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import SGDClassifier

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
import re
from nltk.stem import LancasterStemmer


#df=pd.read_csv('./California_Earthquake_tweets.csv',names=['label','Text'])
#Reading Data and changing annotations to integer labels
df=pd.read_csv('train.csv',names=['label','Text'])
df.loc[df["label"]=='Injured or dead people',"label"]=1
df.loc[df["label"]=='Missing, trapped, or found people',"label"]=1
df.loc[df["label"]=='Displaced people',"label"]=1
df.loc[df["label"]=='Infrastructure and utilities',"label"]=1
df.loc[df["label"]=='Shelter and supplies',"label"]=1
df.loc[df["label"]=='Money',"label"]=0
df.loc[df["label"]=='Animal management',"label"]=0
df.loc[df["label"]=='Caution and advice',"label"]=0
df.loc[df["label"]=='Personal updates',"label"]=0
df.loc[df["label"]=='Sympathy and emotional support',"label"]=0
df.loc[df["label"]=='Other relevant information',"label"]=0
df.loc[df["label"]=='Not related or irrelevant',"label"]=0
df.loc[df["label"]=='Needs of those affected',"label"]=1
df.loc[df["label"]=='Infrastructure',"label"]=1
df.loc[df["label"]=='Donations of money',"label"]=1
df.loc[df["label"]=='Donations of supplies and/or volunteer work',"label"]=1
df.loc[df["label"]=='Personal updates, sympathy, support',"label"]=0
df.loc[df["label"]=='Other useful information',"label"]=1
df.loc[df["label"]=='People missing or found',"label"]=1
df.loc[df["label"]=='Physical landslide',"label"]=1
df.loc[df["label"]=='Not physical landslide',"label"]=0
df.loc[df["label"]=='Informative',"label"]=1
df.loc[df["label"]=='Personal only',"label"]=0
df.loc[df["label"]=='Not related to crisis',"label"]=0
df.loc[df["label"]=='Praying',"label"]=0
df.loc[df["label"]=='Personal',"label"]=0
df.loc[df["label"]=='Non-government',"label"]=0
df.loc[df["label"]=='Yes',"label"]=1
df.loc[df["label"]=='No',"label"]=0
df.loc[df["label"]=='Requests for Help/Needs',"label"]=1
df.loc[df["label"]=='Humanitarian Aid Provided',"label"]=1
df.loc[df["label"]=='Infrastructure Damage',"label"]=1
df.loc[df["label"]=='Other Relevant Information',"label"]=1
df.loc[df["label"]=='Not Informative',"label"]=0
df.loc[df["label"]=='Not informative',"label"]=0
df.loc[df["label"]=='Injured and dead',"label"]=1
df.loc[df["label"]=='Volunteer or professional services',"label"]=1



#Splitting dataset to X & Y
df_x = df["Text"]
df_y = df["label"]

#removing punctuations from data and Stemming
stemmer=LancasterStemmer()
for i in range(len(df_x)):
    words = df_x[i].split()
    df_x[i]=""
    for word in words:
        df_x[i] = df_x[i] + stemmer.stem(word.strip('.@,-()"\'?!#\\:')) + " "

#Removing emoji data
emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    
for i in range(len(df_x)):
    df_x[i] = emoji_pattern.sub(r'', df_x[i])


#Splitting data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(df_x,df_y, test_size=0.2,random_state=4)

#Creating Tfidf Vector and removing stop words
tfidf= TfidfVectorizer(min_df=1,stop_words='english')
x_traintfidf = tfidf.fit_transform(x_train)
x_testtfidf = tfidf.transform(x_test)

#Creating Tfidf Vector and removing stop words
cv= TfidfVectorizer(min_df=1,stop_words='english')
x_traincv = cv.fit_transform(x_train)
x_testcv = cv.transform(x_test)

#Converting the Y training to int
y_train=y_train.astype('int')

#Classifying using Multinomial Naive Bayes
#Training

#Multinomial NB-tfidf
mnb = MultinomialNB()
mnb.fit(x_traintfidf,y_train)
pred1 = mnb.predict(x_testtfidf)

with open('NB_model_tfidf_SmallData.pkl','wb')as fout:
    pickle.dump((tfidf,cv,mnb), fout)

#Multinomial NB-cv
mnb = MultinomialNB()
mnb.fit(x_traincv,y_train)
pred2 = mnb.predict(x_testcv)

with open('NB_model_CV_SmallData.pkl','wb')as fout:
    pickle.dump((tfidf,cv,mnb), fout)

#Classifying using NB_both
text_clf_small_nbboth = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('clf', MultinomialNB())])

_ = text_clf_small_nbboth.fit(x_train, y_train)

pred4 = text_clf_small_nbboth.predict(x_test)


with open('model_text_clf_small_nbboth.pkl','wb')as fout:
    pickle.dump((text_clf_small_nbboth), fout)


#Testing
actual_prediction = np.array(y_test)


count=0
for i in range (len(pred1)):
    if pred1[i]==actual_prediction[i]:
        count=count+1
        
accuracy = count/len(pred1)

print("accuracy for naive_bayes-tfidf = ", accuracy*100,"%")

count=0
for i in range (len(pred2)):
    if pred2[i]==actual_prediction[i]:
        count=count+1
        
accuracy = count/len(pred2)

print("accuracy for naive_bayes-cv = ", accuracy*100,"%")

count=0

for i in range (len(pred4)):
    if pred4[i]==actual_prediction[i]:
        count=count+1
        
accuracy = count/len(pred4)

print("accuracy for nb-both = ", accuracy*100,"%")
