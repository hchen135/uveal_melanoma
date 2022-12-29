from user_study import app
from flask import render_template

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/thank_you')
def thank_you():
    return render_template('thank_you.html')