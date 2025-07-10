from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
import pickle
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

class WorkoutRecommendationSystem:
    def __init__(self, model_path='model_olahraga54.h5', scaler_path='scaler54.pkl', 
                 label_encoder_path='label_encoder54.pkl', mlb_path='mlb54.pkl'):
        """
        Initialize the Workout Recommendation System with trained model and preprocessing components
        """
        try:
            # Load model
            self.model = tf.keras.models.load_model(model_path)
            
            # Load preprocessing components
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
                
            with open(label_encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
                
            with open(mlb_path, 'rb') as f:
                self.mlb = pickle.load(f)
            
            # Store gender mapping for convenience
            self.gender_mapping = dict(enumerate(self.label_encoder.classes_))
            self.reverse_gender_mapping = {v: k for k, v in self.gender_mapping.items()}
            
            # Store recommendation classes
            self.recommendation_classes = self.mlb.classes_
            
            logger.info("Workout Recommendation System initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Workout Recommendation System: {str(e)}")
            raise e
    
    def preprocess_input(self, age, gender, height_cm, weight_kg, bmi=None):
        """
        Preprocess the input data for prediction
        """
        # Calculate BMI if not provided
        if bmi is None:
            bmi = weight_kg / ((height_cm/100) ** 2)
        
        # Convert gender to numerical using LabelEncoder
        if gender in self.reverse_gender_mapping:
            gender_encoded = self.reverse_gender_mapping[gender]
        else:
            raise ValueError(f"Gender must be one of {list(self.gender_mapping.values())}")
        
        # Create input array
        input_data = np.array([[age, gender_encoded, height_cm, weight_kg, bmi]])
        
        # Scale the input data
        scaled_input = self.scaler.transform(input_data)
        
        return scaled_input
    
    def predict(self, age, gender, height_cm, weight_kg, bmi=None, threshold=0.3, top_n=3):
        """
        Make workout recommendations based on user input
        """
        # Preprocess input
        preprocessed_input = self.preprocess_input(age, gender, height_cm, weight_kg, bmi)
        
        # Make prediction
        prediction_probabilities = self.model.predict(preprocessed_input, verbose=0)[0]
        
        # Get recommendations
        recommendations_dict = {self.recommendation_classes[i]: float(prob) 
                               for i, prob in enumerate(prediction_probabilities)}
        
        # Sort by probability
        sorted_recommendations = sorted(recommendations_dict.items(), 
                                       key=lambda x: x[1], reverse=True)
        
        # Get top N recommendations
        top_recommendations = sorted_recommendations[:top_n]
        
        return top_recommendations, recommendations_dict

    def get_bmi_category(self, bmi):
        """
        Get the BMI category based on BMI value
        """
        if bmi < 18.5:
            return "underweight"
        elif bmi < 25:
            return "normal"
        elif bmi < 30:
            return "overweight"
        else:
            return "obese"
    
    def get_age_category(self, age):
        """
        Get age category based on age value
        """
        if age < 18:
            return "Remaja"
        elif age < 35:
            return "Muda"
        elif age < 55:
            return "Dewasa"
        else:
            return "Senior"

# Initialize the recommendation system
try:
    recommender = WorkoutRecommendationSystem()
    model_loaded = True
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    model_loaded = False
    recommender = None

# HTML template for API documentation
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Workout Recommendation API</title>
  <style>
    body {
        font-family: Arial, sans-serif;
        background-color: #fff;
        margin: 0;
        padding: 5px;
    }

    .container {
        max-width: 800px;
        margin: auto;
    }

    .status {
        padding: 10px;
        margin: 5px 0;
    }



    .endpoint {
        margin: 30px 0;
        padding: 5px;
    }

    .method {
        font-weight: bold;
        padding: 3px 6px;
        font-size: 0.9em;
    }

    pre {
        background-color: #eee;
        padding: 5px;
        overflow-x: auto;
        font-family: 'Courier New', monospace;
        font-size: 0.9em;
    }

    ul {
        padding-left: 20px;
    }

    li {
        margin-bottom: 5px;
    }

    p {
        line-height: 1.6;
    }

    

    a {
        color: #000;
        text-decoration: underline;
    }
</style>

</head>
<body>
    <div class="container">
        <h1>Workout Recommendation API</h1>
        <p>Selamat datang di API Rekomendasi Olahraga. Gunakan endpoint berikut untuk mendapatkan rekomendasi olahraga yang dipersonalisasi:</p>
        
        <div class="status {{ 'success' if model_status else 'error' }}">
            <strong>Status Model:</strong> {{ 'Loaded Successfully ' if model_status else 'Failed to Load ❌' }}
        </div>
        
        <div class="endpoint">
            <h2><span class="method get">GET</span>/</h2>
            <p>Menampilkan halaman dokumentasi API ini</p>
        </div>
        
        <div class="endpoint">
            <h2><span class="method get">GET</span>/api/health</h2>
            <p>Mengecek status kesehatan API</p>
            <h3>Response:</h3>
            <pre>{
  "status": "healthy",
  "model_loaded": true,
  "message": "Workout Recommendation API is running"
}</pre>
        </div>
        <div class="endpoint">
            <h2><span class="method post">POST</span>/api/recommend</h2>
            <p>Mendapatkan rekomendasi olahraga yang dipersonalisasi berdasarkan data pribadi</p>
            
            <h3>Request Body:</h3>
            <pre>{
  "age": 30,
  "gender": "Male",
  "height": 175,
  "weight": 70
}</pre>
            
            <h3>Parameters:</h3>
            <ul>
                <li><strong>age</strong> (integer): Umur dalam tahun (1-120)</li>
                <li><strong>gender</strong> (string): Jenis kelamin ("Male" atau "Female")</li>
                <li><strong>height</strong> (number): Tinggi badan dalam cm</li>
                <li><strong>weight</strong> (number): Berat badan dalam kg</li>
            </ul>
            
            <h3>Response:</h3>
            <pre>{
  "success": true,
  "data": {
    "age": 30,
    "gender": "Male",
    "height": 175,
    "weight": 70,
    "bmi": 22.86,
    "bmi_category": "normal",
    "age_category": "Muda",
    "recommended_workouts": [
      {
        "name": "Berenang",
        "confidence": 0.565
      },
      {
        "name": "Berjalan",
        "confidence": 0.563
      }
    ],
    "all_probabilities": {
      "Berenang": 0.565,
      "Berjalan": 0.563,
      "Sepeda": 0.537,
      "Senam": 0.0,
      "Yoga": 0.0,
      "HIIT": 0.0,
      "Weight Training": 0.0,
      "Kardio": 0.0
    }
  }
}</pre>
        </div>
        
        <div class="endpoint">
            <h2>Error Responses</h2>
            <h3>400 Bad Request:</h3>
            <pre>{
  "success": false,
  "error": "Missing required field: age"
}</pre>
            
            <h3>500 Internal Server Error:</h3>
            <pre>{
  "success": false,
  "error": "Model not loaded properly"
}</pre>
        </div>
        
       
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    """Display API documentation"""
    return render_template_string(HTML_TEMPLATE, model_status=model_loaded)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'message': 'Workout Recommendation API is running'
    })

@app.route('/api/recommend', methods=['POST'])
def recommend_workout():
    """Main endpoint for workout recommendations"""
    try:
        # Check if model is loaded
        if not model_loaded or recommender is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded properly'
            }), 500
        
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        # Validate required fields
        required_fields = ['age', 'gender', 'height', 'weight']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Extract and validate data
        age = data['age']
        gender = data['gender']
        height = data['height']
        weight = data['weight']
        
        # Validate data types and ranges
        if not isinstance(age, (int, float)) or age <= 0 or age > 120:
            return jsonify({
                'success': False,
                'error': 'Age must be a number between 1 and 120'
            }), 400
        
        if gender not in ['Male', 'Female']:
            return jsonify({
                'success': False,
                'error': 'Gender must be "Male" or "Female"'
            }), 400
        
        if not isinstance(height, (int, float)) or height <= 0 or height > 300:
            return jsonify({
                'success': False,
                'error': 'Height must be a number between 1 and 300 cm'
            }), 400
        
        if not isinstance(weight, (int, float)) or weight <= 0 or weight > 500:
            return jsonify({
                'success': False,
                'error': 'Weight must be a number between 1 and 500 kg'
            }), 400
        
        # Calculate BMI
        bmi = weight / ((height/100) ** 2)
        
        # Get recommendations
        top_recommendations, all_probabilities = recommender.predict(
            age=age, 
            gender=gender, 
            height_cm=height, 
            weight_kg=weight
        )
        
        # Format recommendations
        formatted_recommendations = [
            {
                'name': workout,
                'confidence': round(confidence, 3)
            }
            for workout, confidence in top_recommendations
        ]
        
        # Format all probabilities
        formatted_probabilities = {
            workout: round(prob, 3)
            for workout, prob in all_probabilities.items()
        }
        
        # Prepare response
        response_data = {
            'success': True,
            'data': {
                'age': age,
                'gender': gender,
                'height': height,
                'weight': weight,
                'bmi': round(bmi, 2),
                'bmi_category': recommender.get_bmi_category(bmi),
                'age_category': recommender.get_age_category(age),
                'recommended_workouts': formatted_recommendations,
                'all_probabilities': formatted_probabilities
            }
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in recommend_workout: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors"""
    return jsonify({
        'success': False,
        'error': 'Method not allowed'
    }), 405

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    print("Starting Workout Recommendation API...")
    print("API Documentation available at: http://localhost:5000/")
    print("Health check available at: http://localhost:5000/api/health")
    print("Recommendation endpoint: http://localhost:5000/api/recommend")
    
    app.run(debug=True, host='0.0.0.0', port=5000)