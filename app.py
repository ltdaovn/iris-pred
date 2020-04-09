from flask import Flask, request, render_template
from joblib import dump, load

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

	
@app.route('/prediction', methods=['POST','GET'])
def prediction():
	if request.method=='POST':
		loaded_model = load('Iris.DecisionTree.joblib')
		classes = ["Setosa", "Versicolor", "Virginica"]
		X_new = ([[request.form['sep_len'], 
			request.form['sep_wid'],
			request.form['pet_len'],
			request.form['pet_wid']]])
		y_new = loaded_model.predict(X_new)
		return render_template('result.html',results=classes[y_new[0]])
	
if __name__ == '__main__':
	app.run()
	