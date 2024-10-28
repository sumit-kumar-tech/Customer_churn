from flask import Flask, render_template, request, flash, redirect, url_for
import pickle
import pandas as pd
import mysql.connector
from mysql.connector import Error

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# MySQL Database Configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'timuS@7269',
    'database': 'churn_db'
}

# Load the machine learning model
model = pickle.load(open('my_model.pkl', 'rb'))

# Map categorical values to numerical data
GENDER_MAP = {'Male': 1, 'Female': 0}
CONTRACT_LENGTH_MAP = {'Annual': 12, 'Quarterly': 3, 'Monthly': 1}

def save_to_database(data):
    """Save form data and prediction to MySQL database."""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        cursor = connection.cursor()

        # Insert data into the predictions table
        query = """
        INSERT INTO predictions (
            age, gender, tenure, usage_frequency, support_calls, 
            payment_delay, contract_length, total_spend, 
            last_interaction, prediction
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(query, data)
        connection.commit()
    except Error as e:
        print(f"Error: {e}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

@app.route('/form')
def home():
    return render_template('form.html', data=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        age = float(request.form['age'])
        gender = request.form['gender']
        tenure = float(request.form['tenure'])
        usage_frequency = float(request.form['usage_frequency'])
        support_calls = float(request.form['support_calls'])
        payment_delay = float(request.form['payment_delay'])
        contract_length = request.form['contract_length']
        total_spend = float(request.form['total_spend'])
        last_interaction = float(request.form['last_interaction'])

        # Prepare the input for prediction
        input_data = pd.DataFrame([[age, gender, tenure, usage_frequency, 
                                    support_calls, payment_delay, 
                                    contract_length, total_spend, last_interaction]],
                                  columns=['Age', 'Gender', 'Tenure', 'Usage Frequency',
                                           'Support Calls', 'Payment Delay',
                                           'Contract Length', 'Total Spend', 'Last Interaction'])
        print(input_data)
        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction_text = "Customer Churn" if prediction == 1 else "Customer Not Churn"

        # Save data to the database
        save_to_database((
            age, gender, tenure, usage_frequency,
            support_calls, payment_delay, contract_length,
            total_spend, last_interaction, prediction_text
        ))

        # Render form with prediction result
        return render_template('form.html', data=[prediction_text, 'red' if prediction == 1 else 'green'])

    except Exception as e:
        flash(f"Error: {str(e)}", "error")
        return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
