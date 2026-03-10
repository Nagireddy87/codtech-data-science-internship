# Task 3 – End-to-End Data Science Project

## Project Title

House Price Prediction API using Flask

## Internship

CodTech Data Science Internship

## Objective

The objective of this project is to build a complete data science pipeline that includes data preprocessing, model training, and deployment using a web API.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Flask

## Project Workflow

1. **Data Collection**

   * House price dataset was used containing features such as bedrooms, bathrooms, and living area.

2. **Data Preprocessing**

   * Selected important features from the dataset.
   * Split the data into training and testing sets.

3. **Model Training**

   * A Linear Regression model was used to train the dataset.
   * The trained model was saved using Pickle.

4. **Model Deployment**

   * A Flask API was created to serve predictions.
   * The API receives input data in JSON format and returns predicted house prices.

## API Endpoint

### Home Route

```
http://127.0.0.1:5000
```

Returns a message confirming that the API is running.

### Prediction Route

```
POST /predict
```

Example JSON Input:

{
"bedrooms": 3,
"bathrooms": 2,
"sqft_living": 1500
}

Example Response:

{
"predicted_price": 450000
}

## Project Structure

Task3_EndToEnd_Project
│
├── train_model.py
├── app.py
├── model.pkl
├── house_data.csv
└── README.md

## Result

The trained machine learning model was successfully deployed using Flask.
Users can send house feature data through the API and receive predicted house prices.

## Author

CodTech Data Science Internship – Task 3
