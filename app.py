from flask import Flask, request, render_template
import numpy as np
import pickle

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model and scaler (make sure you save the trained model)
model = pickle.load(open('house_price_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Home route to display the form
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle the prediction based on user input
@app.route('/predict', methods=['POST'])
def predict():
    # Collect data from form
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    
    # Scale the input features
    scaled_features = scaler.transform(final_features)
    
    # Predict using the trained model
    prediction = model.predict(scaled_features)
    
    # Return the result
    return render_template('index.html', prediction_text='Estimated House Price: ${:.2f}'.format(prediction[0]))

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
