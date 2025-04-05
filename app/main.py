import os
import sys

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from flask import Flask
from routes import setup_routes

app = Flask(__name__)
setup_routes(app)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
