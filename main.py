from flask import Flask
from flask_cors import CORS
from src.controllers.fingerprint_controller import extract_fingerprints

app = Flask(__name__)
CORS(app)

# Routes
app.add_url_rule('/api/v1/fingerprints/extract', 'extract_fingerprints', extract_fingerprints, methods=['POST'])

if __name__ == '__main__':
    app.run(debug=True)
