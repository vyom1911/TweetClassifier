import pickle
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import LancasterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
#Reading Data and changing annotations to integer labels
df=pd.read_csv('test.csv',names=['label','Text'])
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
#Stemming and removing punctuation
stemmer=LancasterStemmer()
for i in range(len(df_x)):
    words = df_x[i].split()
    df_x[i]=""
    for word in words:
        df_x[i] = df_x[i] + stemmer.stem(word.strip('.@,-()?!#\\:\'"')) + " "

#Removing emoji data
emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    
for i in range(len(df_x)):
    df_x[i] = emoji_pattern.sub(r'', df_x[i])

#Load the models and predict the given sample

#Naive Bayes tfidf
with open('NB_model_tfidf_SmallData.pkl', 'rb') as fin:
    tfidf,cv,clf_tf = pickle.load(fin)
X_new = tfidf.transform(df_x)
pred_nbtfidf = clf_tf.predict(X_new)

#Naive Bayes cv
with open('NB_model_CV_SmallData.pkl', 'rb') as fin:
    tfidf,cv,clf_cv = pickle.load(fin)
X_new = cv.transform(df_x)
pred_nbcv = clf_cv.predict(X_new)

#Naive Bayes Both pipelined
with open('NB_model_both_SmallData.pkl', 'rb') as fin:
    clf = pickle.load(fin)
pred_nbboth = clf.predict(df_x)
	
#SVM tfidf
with open('model_svm_small_tfidf.pkl', 'rb') as fin:
    clf = pickle.load(fin)
pred_svmtfidf = clf.predict(df_x)
	
#SVM cv
with open('model_svm_small_cv.pkl', 'rb') as fin:
    clf = pickle.load(fin)
pred_svmcv = clf.predict(df_x)
	
#SVM both
with open('model_svm_small_both.pkl', 'rb') as fin:
    clf = pickle.load(fin)
pred_svmboth = clf.predict(df_x)
		
#Testing and count True Positive (TP)...
actual_prediction = np.array(df_y)

#List to store tweets
tweet_indexes =list() 

#Naive Bayes tf-idf
TP=0
FP=0
FN=0
TN=0
print("Naive Bayes tfidf:")
for i in range (len(df_y)):
    if actual_prediction[i]==1:
        if pred_nbtfidf[i]==actual_prediction[i]:
            TP=TP+1
            tweet_indexes.append(i)
        else:
            FN=FN+1
    else:
        if pred_nbtfidf[i]==actual_prediction[i]:
            TN=TN+1
        else:
            FP=FP+1    
accuracy = (TP+TN)/len(df_y)

print("True Positive = ", TP)
print("False Negative = ", FN)
print("True Negative = ", TN)
print("False Positive = ", FP)
print("accuracy = ", accuracy*100,"%\n")

#Naive Bayes cv
TP=0
FP=0
FN=0
TN=0
print("Naive Bayes count vectorization: ")
for i in range (len(df_y)):
    if actual_prediction[i]==1:
        if pred_nbcv[i]==actual_prediction[i]:
            TP=TP+1
            tweet_indexes.append(i)
        else:
            FN=FN+1
    else:
        if pred_nbcv[i]==actual_prediction[i]:
            TN=TN+1
        else:
            FP=FP+1    
        
accuracy = (TP+TN)/len(df_y)

print("True Positive = ", TP)
print("False Negative = ", FN)
print("True Negative = ", TN)
print("False Positive = ", FP)
print("accuracy = ", accuracy*100,"%\n")

#Naive Bayes both pipelined
TP=0
FP=0
FN=0
TN=0
print("Naive Bayes tfidf & count vectorizer combined: ")
for i in range (len(df_y)):
    if actual_prediction[i]==1:
        if pred_nbboth[i]==actual_prediction[i]:
            TP=TP+1
            tweet_indexes.append(i)
        else:
            FN=FN+1
    else:
        if pred_nbboth[i]==actual_prediction[i]:
            TN=TN+1
        else:
            FP=FP+1    
        
accuracy = (TP+TN)/len(df_y)

print("True Positive = ", TP)
print("False Negative = ", FN)
print("True Negative = ", TN)
print("False Positive = ", FP)
print("accuracy = ", accuracy*100,"%\n")

#SVM tf idf
TP=0
FP=0
FN=0
TN=0
print("SVM tfidf: ")
for i in range (len(df_y)):
    if actual_prediction[i]==1:
        if pred_svmtfidf[i]==actual_prediction[i]:
            TP=TP+1
            tweet_indexes.append(i)
        else:
            FN=FN+1
    else:
        if pred_svmtfidf[i]==actual_prediction[i]:
            TN=TN+1
        else:
            FP=FP+1    
        
accuracy = (TP+TN)/len(df_y)

print("True Positive = ", TP)
print("False Negative = ", FN)
print("True Negative = ", TN)
print("False Positive = ", FP)
print("accuracy = ", accuracy*100,"%\n")

#SVM CV
TP=0
FP=0
FN=0
TN=0
print("SVM Count Vectorizer: ")
for i in range (len(df_y)):
    if actual_prediction[i]==1:
        if pred_svmcv[i]==actual_prediction[i]:
            TP=TP+1
            tweet_indexes.append(i)
        else:
            FN=FN+1
    else:
        if pred_svmcv[i]==actual_prediction[i]:
            TN=TN+1
        else:
            FP=FP+1    
        
accuracy = (TP+TN)/len(df_y)

print("True Positive = ", TP)
print("False Negative = ", FN)
print("True Negative = ", TN)
print("False Positive = ", FP)
print("accuracy = ", accuracy*100,"%\n")

#SVM both
TP=0
FP=0
FN=0
TN=0
print("SVM Both: ")
for i in range (len(df_y)):
    if actual_prediction[i]==1:
        if pred_svmboth[i]==actual_prediction[i]:
            TP=TP+1
            tweet_indexes.append(i)
        else:
            FN=FN+1
    else:
        if pred_svmboth[i]==actual_prediction[i]:
            TN=TN+1
        else:
            FP=FP+1    
        
accuracy = (TP+TN)/len(df_y)

print("True Positive = ", TP)
print("False Negative = ", FN)
print("True Negative = ", TN)
print("False Positive = ", FP)
print("accuracy = ", accuracy*100,"%\n")

em_tweets= set(tweet_indexes)
print("Total Emergency Tweets identified: ",len(em_tweets),"\n\n Tweets saved to emergency_tweets.csv")

emergency_tweets=list()
dataset=pd.read_csv('test.csv',names=['label','Text'])

for i in em_tweets:
    emergency_tweets.append(dataset['Text'][int(i)])

emergency= pd.DataFrame(np.array(emergency_tweets))
emergency.to_csv('emergency_tweets.csv',encoding='utf-8')