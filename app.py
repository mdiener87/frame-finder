#!/usr/bin/env python3
"""
Main application entry point.
This file wraps the new API implementation to maintain compatibility with 'flask run'.
"""

# Import the new API implementation
from api import app

if __name__ == '__main__':
    app.run(debug=True)