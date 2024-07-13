# app.py

from flask import Flask
from dotenv import load_dotenv
import os

app = Flask(__name__)


@app.route('/')
def hello():
    return "Hello, World!"


if __name__ == '__main__':
    app.run(debug=True)
