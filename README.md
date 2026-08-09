NYC Airbnb Room Type Prediction

A full-stack machine learning application that predicts the room typeof an NYC Airbnb listing from listing, location, pricing,availability, review, and host information.

The project covers the complete workflow:

Data → EDA → Cleaning → Preprocessing → Model Comparison →Hyperparameter Tuning → Model Serialization → FastAPI → Web Frontend →Deployment

Demo

Backend API

API: https://nyc-airbnb-room-prediction.onrender.com

Swagger documentation:https://nyc-airbnb-room-prediction.onrender.com/docs

Frontend

Live application: https://nyc-airbnb-room-prediction-1.onrender.com/

Project Overview

The model predicts one of three Airbnb room types:

Entire home/apt

Private room

Shared room

The training notebook uses the New York City Airbnb Open Datadataset from Kaggle.

The model uses listing attributes including:

Latitude

Longitude

Price

Minimum nights

Number of reviews

Reviews per month

Calculated host listings count

Availability for 365 days

Neighbourhood group

Neighbourhood

The complete preprocessing pipeline is serialized together with thetrained classifier so that the API can make predictions without manuallyreproducing preprocessing steps.

Features

Machine Learning

Exploratory Data Analysis (EDA)

Missing-value handling

Outlier treatment

Feature selection

Numerical preprocessing

Categorical preprocessing

Train/test split with stratification

Multiple classification algorithms

3-fold stratified cross-validation

Macro-F1 evaluation for imbalanced classes

Randomized hyperparameter search

Confusion matrix evaluation

Feature/model pipeline serialization

Backend

FastAPI REST API

Pydantic request validation

CORS support

Scikit-learn pipeline inference

Prediction probabilities

Swagger/OpenAPI documentation

Frontend

Responsive HTML/CSS/JavaScript interface

Listing input form

Example listing button

API health indicator

Prediction result visualization

Probability bars

Animated NYC-style building visualization

Loading state and error handling

Reduced-motion support

Machine Learning Workflow

1. Dataset

The project uses the New York City Airbnb Open Data dataset.

Target:

room_type

Target classes:

Entire home/apt
Private room
Shared room

The dataset contains location, pricing, reviews, host, availability, androom-type information.

2. Data Cleaning

The notebook removes fields considered identifiers, free text, orotherwise unsuitable for the tabular model:

id
name
host_id
host_name
last_review

reviews_per_month missing values are replaced with 0, representinglistings without review activity.

Extreme values in price and minimum_nights are capped at their 99thpercentile rather than deleting the affected rows.

3. Features

The final model uses:

Numerical features

latitude
longitude
price
minimum_nights
number_of_reviews
reviews_per_month
calculated_host_listings_count
availability_365

Categorical features

neighbourhood_group
neighbourhood

4. Preprocessing Pipeline

A ColumnTransformer is used so preprocessing and model inferenceremain consistent.

Numerical pipeline

SimpleImputer(strategy="median")
        ↓
StandardScaler()

Categorical pipeline

SimpleImputer(strategy="most_frequent")
        ↓
OneHotEncoder(handle_unknown="ignore")

The preprocessor is combined with the classifier inside a singleScikit-learn Pipeline.

Model Comparison

The notebook compares:

Logistic Regression

Decision Tree

Random Forest

Gradient Boosting

The recorded 3-fold cross-validation results are:

Model                   Accuracy   Macro-F1

Logistic Regression        0.659      0.522Decision Tree              0.782      0.647Random Forest              0.851      0.715Gradient Boosting          0.850      0.705

Random Forest produced the strongest cross-validation Macro-F1 among thetested models.

Because the target classes are imbalanced, Macro-F1 is used alongsideaccuracy so that minority classes are not ignored.

Hyperparameter Tuning

Random Forest is tuned with RandomizedSearchCV.

Search space:

{
    "classifier__n_estimators": [100, 200, 150, 300],
    "classifier__max_depth": [8, 12, 15, 20, None],
    "classifier__min_samples_split": [2, 5, 10],
}

The recorded best parameters are:

n_estimators = 200
min_samples_split = 10
max_depth = None

Best cross-validation Macro-F1:

0.72998

Final Model Performance

On the held-out test set, the recorded final results are:

Accuracy: 0.85591
Macro-F1: 0.74104

The notebook also evaluates the final model using a confusion matrix.

These numbers describe the recorded notebook run. If the dataset,preprocessing, random seed, or scikit-learn version changes, theresults may change.

Model Artifact

The trained preprocessing + classification pipeline is saved as:

Model_Pipeline.pkl

It contains both:

Preprocessing
     +
Random Forest classifier

This allows the FastAPI application to send raw listing featuresdirectly to the trained pipeline.

The current FastAPI application loads:

model = joblib.load("Model_Pipeline.pkl")

Backend API

The API is built with FastAPI.

Health Check

GET /

Example response:

{
  "message": "NYC aribnb room prediction API"
}

Prediction

POST /predict

Request body:

{
  "latitude": 40.7484,
  "longitude": -73.9857,
  "price": 120,
  "minimum_nights": 2,
  "number_of_reviews": 84,
  "reviews_per_month": 2.3,
  "calculated_host_listings_count": 1,
  "availability_365": 210,
  "neighbourhood_group": "Manhattan",
  "neighbourhood": "Midtown"
}

Response structure:

{
  "predicted_room_type": "Entire home/apt",
  "probability": [0.80, 0.18, 0.02]
}

The probability array follows the model's class ordering.

Input Validation

The FastAPI/Pydantic model validates:

Feature                          Validation

latitude                         -90 to 90longitude                        -180 to 180price                            greater than 0minimum_nights                   1 to 365number_of_reviews                0 or greaterreviews_per_month                0 or greatercalculated_host_listings_count   0 or greateravailability_365                 0 to 365neighbourhood_group              non-empty stringneighbourhood                    non-empty string

Invalid requests return a validation error instead of being sentdirectly to the model.

Frontend

The frontend is a static HTML/CSS/JavaScript application.

Main files

index.html
style.css
script.js

The JavaScript frontend sends prediction requests to:

https://nyc-airbnb-room-prediction.onrender.com/predict

It also checks the backend health endpoint and displays either:

API connected

or

API unreachable

The interface includes three sample listings that can be loaded with theTry an example button.

Project Structure

NYC_AIRBNB_ROOM_PREDICTION/
│
├── .venv/
│
├── .gitignore
├── AB_NYC_2019.csv
│
├── index.html
├── style.css
├── script.js
│
├── main.py
│
├── Model_Pipeline.pkl
├── room_model_pipeline.pkl
│
├── nyc_airbnb_room_type_classification.ipynb
│
├── project.html
├── requirements.txt
├── runtime.txt
└── README.md

Important files

File                                          Purpose

main.py                                     FastAPI backend and predictionendpoint

Model_Pipeline.pkl                          Trained Scikit-learn pipeline usedby the API

nyc_airbnb_room_type_classification.ipynb   ML training and evaluation notebook

index.html                                  Frontend structure

style.css                                   Frontend styling

script.js                                   API communication and resultvisualization

AB_NYC_2019.csv                             NYC Airbnb dataset

requirements.txt                            Python dependencies

Installation

1. Clone the repository

git clone https://github.com/AnujrajShrestha/nyc-airbnb-room-prediction.git
cd nyc-airbnb-room-prediction

2. Create a virtual environment

Windows:

python -m venv .venv

Activate it:

.venv\Scripts\activate

3. Install dependencies

python -m pip install -r requirements.txt

If you are only running the API and do not need to retrain the model,the required backend packages include FastAPI, Uvicorn, Joblib,Pydantic, Pandas, and Scikit-learn.

Run the FastAPI Backend

Start the development server:

python -m uvicorn main:app --reload

The API should be available at:

http://127.0.0.1:8000

Swagger documentation:

http://127.0.0.1:8000/docs

Run the Frontend Locally

Because the frontend is static, it can be served using a simple localweb server.

For example:

python -m http.server 5500

Then open:

http://127.0.0.1:5500

The frontend's API_BASE_URL in script.js currently points to thedeployed Render API.

For local development, change it to:

const API_BASE_URL = "http://127.0.0.1:8000";

Deployment

The current project is deployed using Render.

Backend

The FastAPI service is deployed at:

https://nyc-airbnb-room-prediction.onrender.com

Frontend

The frontend is deployed at:

https://nyc-airbnb-room-prediction-1.onrender.com/

The frontend communicates with the backend through the /predictendpoint.

API Architecture

                    ┌─────────────────────────┐
                    │      Web Frontend       │
                    │ HTML + CSS + JavaScript │
                    └────────────┬────────────┘
                                 │
                                 │ POST /predict
                                 ▼
                    ┌─────────────────────────┐
                    │       FastAPI API       │
                    │      main.py             │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Pydantic Validation     │
                    │      Features           │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Model_Pipeline.pkl      │
                    │ Preprocessor + RF Model │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Room Type Prediction    │
                    │ + Probabilities         │
                    └─────────────────────────┘

Machine Learning Architecture

Raw Airbnb Data
      │
      ▼
Data Cleaning
      │
      ├── Remove identifiers/free-text fields
      ├── Fill missing reviews_per_month
      └── Clip extreme price/minimum-night values
      │
      ▼
Train / Test Split
      │
      ▼
ColumnTransformer
      │
      ├── Numerical
      │      ├── Median Imputation
      │      └── StandardScaler
      │
      └── Categorical
             ├── Most-Frequent Imputation
             └── One-Hot Encoding
      │
      ▼
Random Forest Classifier
      │
      ▼
RandomizedSearchCV
      │
      ▼
Final Pipeline
      │
      ▼
Model_Pipeline.pkl

Known Implementation Notes

1. Notebook split description vs code

The notebook's written section describes a 20% test holdout, but thecurrent executable code uses:

train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=42,
    stratify=y
)

Therefore, the current recorded model results are based on a 33% testsplit.

2. Frontend response keys

The FastAPI response model currently returns:

{
  "predicted_room_type": "...",
  "probability": [...]
}

The current script.js result-rendering code should use those exactlowercase keys when reading the API response.

If your frontend currently uses:

result.Predicted_room_type
result.Probability

change them to:

result.predicted_room_type
result.probability

Otherwise the prediction may not render even when the API requestsucceeds.

3. Model files

The repository contains both:

Model_Pipeline.pkl
room_model_pipeline.pkl

The current main.py loads:

Model_Pipeline.pkl

Keep the deployed filename and the filename referenced in main.pysynchronized.

Future Improvements

Possible next steps:

Add probability calibration

Improve minority-class performance for Shared room

Add class-specific precision/recall/F1 reporting

Add automated API tests with Pytest

Add request/response schemas to a dedicated module

Add Docker support

Add CI/CD with GitHub Actions

Add model versioning

Add model monitoring and logging

Add a prediction history database

Add more interactive geographic visualizations

Add automated model retraining

Add unit and integration tests

Improve frontend accessibility and mobile UX

Technologies Used

Machine Learning

Python

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

Joblib

Kaggle dataset

Backend

FastAPI

Pydantic

Uvicorn

Pandas

Joblib

Scikit-learn

CORS middleware

Frontend

HTML5

CSS3

JavaScript

Google Fonts

Deployment

Render

Learning Outcomes

This project demonstrates practical machine-learning deployment ratherthan only model training.

Key concepts demonstrated:

End-to-end supervised classification

Handling imbalanced classes

Cross-validation

Macro-F1 evaluation

Feature preprocessing with ColumnTransformer

Reusable Scikit-learn pipelines

Hyperparameter optimization

Model serialization with Joblib

REST API development with FastAPI

Pydantic validation

Frontend-to-backend integration

CORS configuration

Production-style model inference

Cloud deployment

Author

Anuj Shrestha

GitHub: https://github.com/AnujrajShrestha/

License

Add a project-specific license here if you intend to distribute therepository publicly.~