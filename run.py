#!/usr/bin/env python3
"""
Development server entrypoint for YouTube Wrapped API.

Usage:
    export FLASK_APP=youtube_wrapped.routes:create_app
    flask run

Or:
    python run.py
"""

import os

from youtube_wrapped.routes import create_app

app = create_app()

if __name__ == "__main__":
    # macOS often binds port 5000 to AirPlay Receiver; 5050 is a safe default.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5050")), debug=True)
