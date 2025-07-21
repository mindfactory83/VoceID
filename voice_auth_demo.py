from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os
import base64
import io
import soundfile as sf
from scipy.io import wavfile
import pickle
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app)

# Simulated user database (in-memory for demo)
user_database = {}
model = RandomForestClassifier(n_estimators=100)

# Function to extract MFCC features
def extract_mfcc(audio_data, sample_rate=16000):
    try:
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
        return np.mean(mfcc.T, axis=0)  # Return mean MFCC features
    except Exception as e:
        return None, f"MFCC extraction failed: {str(e)}"

# Function to generate MFCC visualization
def generate_mfcc_plot(audio_data, sample_rate=16000):
    try:
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfcc, x_axis='time')
        plt.colorbar()
        plt.title('MFCC')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.json
        user_id = data['user_id']
        audio_data = base64.b64decode(data['audio'].split(',')[1])
        
        # Convert base64 audio to numpy array
        audio_file = io.BytesIO(audio_data)
        audio, sample_rate = sf.read(audio_file)
        
        # Extract MFCC features
        mfcc_features = extract_mfcc(audio, sample_rate)
        if mfcc_features is None:
            return jsonify({'error': 'Failed to extract MFCC features'}), 400
            
        # Generate MFCC visualization
        mfcc_plot = generate_mfcc_plot(audio, sample_rate)
        
        # Store user data
        user_database[user_id] = {
            'mfcc': mfcc_features,
            'audio': audio_data
        }
        
        # Train model if we have enough users
        if len(user_database) > 1:
            X = np.array([user['mfcc'] for user in user_database.values()])
            y = list(user_database.keys())
            model.fit(X, y)
            
        return jsonify({
            'message': 'User registered successfully',
            'mfcc_plot': mfcc_plot,
            'mfcc_features': mfcc_features.tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/authenticate', methods=['POST'])
def authenticate():
    try:
        data = request.json
        audio_data = base64.b64decode(data['audio'].split(',')[1])
        
        # Convert base64 audio to numpy array
        audio_file = io.BytesIO(audio_data)
        audio, sample_rate = sf.read(audio_file)
        
        # Extract MFCC features
        mfcc_features = extract_mfcc(audio, sample_rate)
        if mfcc_features is None:
            return jsonify({
                'error': 'Failed to extract MFCC features',
                'reason': 'Audio processing error',
                'improvement': 'Ensure clear audio input with minimal background noise'
            }), 400
            
        # Generate MFCC visualization
        mfcc_plot = generate_mfcc_plot(audio, sample_rate)
        
        if len(user_database) == 0:
            return jsonify({
                'error': 'No registered users',
                'reason': 'Database is empty',
                'improvement': 'Register users first'
            }), 400
            
        # Predict user
        prediction = model.predict([mfcc_features])[0]
        confidence = model.predict_proba([mfcc_features])[0].max()
        
        if confidence < 0.7:  # Arbitrary threshold
            return jsonify({
                'authenticated': False,
                'reason': 'Low confidence score',
                'confidence': float(confidence),
                'mfcc_plot': mfcc_plot,
                'improvement': 'Try speaking more clearly or in a quieter environment'
            })
            
        return jsonify({
            'authenticated': True,
            'user_id': prediction,
            'confidence': float(confidence),
            'mfcc_plot': mfcc_plot
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'reason': 'Processing error',
            'improvement': 'Check audio format and try again'
        }), 500

if __name__ == '__main__':
    app.run(debug=True)