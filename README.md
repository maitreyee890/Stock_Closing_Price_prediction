# Stock_Closing_Price_prediction
## README

**Project Title:**  
Stock Price Prediction Using Machine Learning and Deep Learning Models

---

**Overview:**  
This project analyzes and predicts stock prices using a combination of statistical, machine learning, and deep learning models. The workflow includes data preprocessing, feature engineering, and the implementation of regression, Random Forest, and LSTM neural network models to forecast closing prices and evaluate their performance using standard metrics.

---

## **Table of Contents**
- Project Structure
- Data Preparation
- Feature Engineering
- Model Building
- Model Evaluation
- Requirements
- How to Run
- Results Summary
- Author

---

## **Project Structure**
- `Maitreyee_Maheshwari.ipynb` — Main Jupyter Notebook containing all code, data handling, and results.
- Data is assumed to be provided in a CSV or DataFrame format with columns: `Date`, `Open`, `High`, `Low`, `Close`.

---

## **Data Preparation**
- The dataset consists of monthly stock prices with columns: Date, Open, High, Low, Close.
- The `Date` column is parsed as a datetime object.
- Missing values are checked and confirmed to be absent in all columns.
- Additional features such as `Year`, `Month`, and `Day` are extracted from the Date column for enhanced modeling.

---

## **Feature Engineering**
- Lag features (e.g., `Close_Lag1`) and rolling statistics (e.g., `Rolling_Mean_3`) are generated to capture temporal dependencies.
- The final feature set includes:  
  `Date`, `Open`, `High`, `Low`, `Close`, `Year`, `Month`, `Day`, `Close_Lag1`, `Rolling_Mean_3'.

---

## **Model Building**
Three main models are implemented:

- **Linear Regression:**  
  A baseline statistical model for regression analysis.

- **Random Forest Regressor:**  
  An ensemble model to capture non-linear relationships and improve prediction robustness.

- **LSTM Neural Network:**  
  A deep learning sequence model with the following architecture:
  - LSTM Layer (60 timesteps, 50 units)
  - LSTM Layer (50 units)
  - Dense Layer (25 units)
  - Output Dense Layer (1 unit)
  - Total trainable parameters: 31,901

---

## **Model Training**
- The LSTM model is trained for 100 epochs with a batch size of 32.
- Validation is performed on a held-out test set.
- Training and validation loss are monitored throughout the epochs.

---

## **Model Evaluation**
- Predictions are made on the test set.
- Performance metrics include:
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)
  - R² Score

---

## **Requirements**
pandas
numpy
matplotlib
scikit-learn
keras
tensorflow
yfinance (if live data fetching is used)
seaborn (for advanced plotting)
datetime (standard library, for date handling)

---

## **How to Run**
1. Open `Maitreyee_Maheshwari.ipynb` in Jupyter Notebook or Google Colab.
2. Ensure all required libraries are installed.
3. Run all cells sequentially to preprocess the data, engineer features, train models, and evaluate results.
4. Modify file paths or parameters as needed for your dataset.

---

## **Results Summary**
- The notebook demonstrates end-to-end stock price prediction using multiple modeling approaches.
- Feature engineering and model selection are critical for improving performance.
- LSTM and Random Forest models are compared against baseline regression, with metrics reported for each[1][2][3].

---

