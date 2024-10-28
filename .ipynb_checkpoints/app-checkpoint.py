from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

@app.route('/form')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = float(request.form['age'])
        gender = request.form['gender']
        tenure = float(request.form['tenure'])
        usage_frequency = float(request.form['usage_frequency'])
        support_calls = float(request.form['support_calls'])
        payment_delay = float(request.form['payment_delay'])
        contract_length = request.form['contract_length']
        total_spend = float(request.form['total_spend'])
        last_interaction = float(request.form['last_interaction'])

        result=[age,gender,tenure,usage_frequency,support_calls,payment_delay,contract_length,total_spend,last_interaction]
        print(result)


        feature = ['Age', 'Gender', 'Tenure', 'Usage Frequency', 'Support Calls',
       'Payment Delay', 'Contract Length', 'Total Spend', 'Last Interaction']
        
        res = dict(map(lambda i,j : (i,[j]) , feature,result))

        test_data = pd.DataFrame(res)
        print(test_data)


        model = pickle.load(open('my_model.pkl','rb'))
        prediction = model.predict(test_data)
        print(prediction) 



        if prediction[0] == 1:
            return render_template('form.html',data=["Customer Churn",'blue'])
        else:
            return render_template('form.html',data=['Customer not churn','green'])
    else:
        return "something went wrong"

if __name__ == '__main__':
    app.run(debug=True)  