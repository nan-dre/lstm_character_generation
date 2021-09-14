from flask import Flask, Markup
from generate import generate
import re

app = Flask(__name__)

@app.route("/")
def hello_world():
    text = generate()
    text = re.sub(r'\n', r'<br>', text)
    print(text)
    return text