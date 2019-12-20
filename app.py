from flask import Flask, render_template, request
import os
from trainingBot import *

app = Flask(__name__)
@app.route('/home') 
def home():                     
    return render_template("home.html")


@app.route("/get")
def get_bot_response():       
	if not os.path.isfile('mymodel.pt'):
		print(os.path.isfile('mymodel.pt'))
		train()
	model = torch.load('mymodel.pt')
	model.eval()

	inp = request.args.get('msg')
	return response(inp, model)
    	

if __name__ == "__main__":
	app.run()