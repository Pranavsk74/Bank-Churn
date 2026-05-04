# This file is used by Vercel to start the Flask application
import sys
import os

# Ensure the root directory is in the path
sys.path.insert(0, os.path.dirname(__file__))

from app.backend import app

if __name__ == "__main__":
    app.run(debug=True)
