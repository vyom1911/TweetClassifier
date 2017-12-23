#!/Program Files/Anaconda3/python.exe
import cgi, cgitb 
import pickle
import pandas as pd
import numpy as np
import re
#from nltk.stem import LancasterStemmer
# Create instance of FieldStorage 
form = cgi.FieldStorage() 

# Get data from fields
tweet = form.getvalue('param')
tweet=str(tweet)

#Creating a dataframe of inputs
df = pd.DataFrame({'Text': [tweet]})
#separating column
df_x = df["Text"]

#Removing Punctuation and Stemming
#stemmer=LancasterStemmer()

for i in range(len(df_x)):
    words = df_x[i].split()
    df_x[i]=""
    for word in words:
        df_x[i] = df_x[i] + word.strip('.@,-()"\'?!#\\:') + " "

#Removing emoji data
emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
for i in range(len(df_x)):
    df_x[i] = emoji_pattern.sub(r'', df_x[i])

#Opening our saved model

with open('model_svm_small_cv.pkl', 'rb') as fin:
    cv = pickle.load(fin)
    

X_new_preds = cv.predict(df_x)

if(X_new_preds[0]==0):
    print("Content-Type: text/html\n\n")
    print ("Non-emergency Tweet")
else:
    print("Content-Type: text/html\n\n")
    print ("Emergency Tweet")