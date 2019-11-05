from flask import Flask,render_template,url_for,request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/prediction',methods=['GET','POST'])

def prediction():

	#loading data from csv file

	import pandas as pd
	dataset=pd.read_csv("dataset_new.csv", encoding="latin-1")
	dataset.head()

	#labels
	y=dataset["label"]

	#features
	urls=dataset["url"]

	#feature extraction using Tfidfvectorizer

	from sklearn.feature_extraction.text import TfidfVectorizer
	tfidfv = TfidfVectorizer()

	#data fitting and transform

	x=tfidfv.fit_transform(urls)

	#spliting dataset into training and testing datasets 

	from sklearn.model_selection import train_test_split
	x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=34)
	

	#model buiding 

	from sklearn.naive_bayes import MultinomialNB
	implement=MultinomialNB()
	implement.fit(x_train,y_train)
	
	#accuracy of the model

	#print(implement.score(x_test,y_test))
	if request.method == 'POST':  # take input from the html form as POST request method
		message = request.form['message']
		data = [message]
		new= tfidfv.transform(data).toarray()
		my_prediction = implement.predict(new)

	return render_template('output.html',predict = my_prediction)	

if __name__ == '__main__':
	app.run()

