from flask import render_template
from flaskexample import app
#from sqlalchemy import create_engine
#from sqlalchemy_utils import database_exists, create_database
import pandas as pd
#import psycopg2
from flask import request
from flaskexample.a_Model import ModelIt
import flaskexample.classify
DEFAULT = flaskexample.classify.DEFAULT

# Python code to connect to Postgres
# You may need to modify this based on your OS,
# as detailed in the postgres dev setup materials.
#user = 'ziyingfeng' #add your Postgres username here
#host = 'localhost'
#dbname = 'birth_db'
#db = create_engine('postgres://%s%s/%s'%(user,host,dbname))
#con = None
#con = psycopg2.connect(database = dbname, user = user)

@app.route('/')
@app.route('/index')
def index():
    STYLESHEET = render_template("styles.css", output=DEFAULT)
    return render_template("index.html", output=DEFAULT, STYLESHEET=STYLESHEET)

@app.route('/', methods=['POST'])
def results():
    link = request.form['InputLink']
    if link == '':
        STYLESHEET = render_template("styles.css", output=DEFAULT)
        return render_template("index.html", output=DEFAULT, STYLESHEET=STYLESHEET)
    output = flaskexample.classify.classify(link)
    STYLESHEET = render_template("styles.css", output=output)
    return render_template("index.html", output=output, STYLESHEET=STYLESHEET)
