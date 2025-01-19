from flask import Flask, render_template, request
import pandas as pd
import pickle 

app = Flask(__name__)
data = pd.read_csv('cleaned_data.csv')
pipe = pickle.load(open('model.pkl', 'rb'))

@app.route("/")
def index():

  locations = sorted(data['location'].unique())
  area_type = sorted(data['area_type'].unique())
  availability = sorted(data['availability'].unique())

  return render_template('index.html', 
                         locations= locations, 
                         area = area_type, 
                         availability= availability)

@app.route('/predict', methods=['POST'])
def predict():
  area = request.form.get('area')
  availability = request.form.get('availability')
  location = request.form.get('location')
  bhk = float(request.form.get('bhk'))
  bath = float(request.form.get('bath'))
  sqft = float(request.form.get('sqft'))


  input = pd.DataFrame([[area, availability, location, bhk, sqft, bath]],
                       columns=['area_type', 'availability', 'location', 'size_bhk', 'total_sqft',
       'bath'])
  
  predictions = pipe.predict(input)[0]
  return str(predictions.round(2))

if __name__ == "__main__":
  app.run(debug=True, port=5001)