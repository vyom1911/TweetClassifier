# TweetClassifier
This project classifies tweets into emergency and non-emergency tweets using SVM and Multinomial Naive Bayes which can be useful to identify people asking for help on twitter during crisis

## Instruction to run project:

To run our python application:
1) Install necessary dependencies like Scikit learn, NLTK and Pandas
2) In the apply.py file add path to your location
3) Execute the apply.py file

To run our Web Application:
1) You will need a WAMP server or equivalent server to host our Web Application.
2) In WAMP, In http.conf file look for this line: AddHandler cgi-script .cgi .pl .asp. Modify it so it looks like this: AddHandler cgi-script .cgi .pl .asp .py
3) You need CGI dependency package for Python
4) Copy our Web App to htdocs folder
5) In classifier.py update the path to the model you want to use. Models are in our zip file
6) Run the web app on localhost

## Dataset Citations:
We used the dataset from the crisisNLP research which belongs to paper:-
Muhammad Imran , Prasenjit Mitra , Carlos Castillo; Twitter as a Lifeline: Human-annotated Twitter
Corpora for NLP of Crisis-related Messages; ​ In Proceedings of the 10th Language Resources and
Evaluation Conference (LREC), pp. 1638-1643. May 2016, Portorož, Slovenia.

Dataset Link: http://crisisnlp.qcri.org/lrec2016/lrec2016.html
