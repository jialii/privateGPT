from flask import Flask,jsonify, render_template, flash, redirect, url_for, Markup, request, session
from flask_cors import CORS
from newFile import model
import os, json
app = Flask(__name__)
CORS(app)

@app.route('/')
def hello():
    data = "hello world"
    return render_template('index.html', posts=data)

@app.route('/demo')
def demo():
    return render_template('chat.html')

@app.route('/get_answer', methods=['GET','POST'])
def get_answer():
    query = request.form['message']
    query = request.form.get('message')
    print(query)
    print(os.environ.get('MODEL_TYPE'))
    # return query
    while query != 'exit':
        query = request.form.get('message')
        if query not in ['', None]:
            response = list(model("You must act as the documents in the knowledge base. Never refer to anything outside of the documents you are trained on. Do not give examples Answer the following question once: "+query))
            if "Helpful Answer" in response[0]:
                answer = response[0].split("\n")[0]
            else:
                answer = response[0]
            time = response[1]
            # print("hereeeee: \n", answer, time)

            # print("response:")
            # print(response)
        
        #can add source data used
        # return jsonify(query=query,answer=answer,time=time)
        # return render_template('index.html', query=query,answer=answer,time=time)
        return jsonify(answer=answer,time=time)
    return 'empty query'


if __name__ == "__main__":
  app.run(host="0.0.0.0", debug = True)