#!/Program Files/Anaconda3/python.exe
import cgi, cgitb 
import pickle
import pandas as pd
import numpy as np
# Create instance of FieldStorage 
form = cgi.FieldStorage() 

# Get data from fields
tweet = form.getvalue('param')
tweet=str(tweet)

#Creating a dataframe of inputs
df = pd.DataFrame({'Text': [tweet]})
#separating column
df_x = df["Text"]

#Opening our saved model
with open('model.pkl', 'rb') as fin:
    tfidf,cv,clf = pickle.load(fin)
    
X_new = cv.transform(df_x)
X_new_preds = clf.predict(X_new)
    
if(X_new_preds[0]==0):
    print("Content-Type: text/html\n\n")
    print ("Non-emergency Tweet")
else:
    print("Content-Type: text/html\n\n")
    print ("Emergency Tweet")