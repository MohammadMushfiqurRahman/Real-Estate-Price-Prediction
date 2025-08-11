# Real Estate Price Prediction

This repository contains a machine learning project that predicts housing prices using various features. The model is trained on a real estate dataset and utilises the RandomForestRegressor from scikit-learn to make predictions and used XAI to understand the models.

## Project Overview

The objective of this project is to build a machine learning model that can predict real estate prices based on several input features like crime rate, number of rooms, and other housing characteristics. The project includes steps such as data cleaning, feature engineering, model training, and evaluation.

## Requirements

To run this project, you'll need the following libraries:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `shap`
- `joblib`

You can install the required dependencies using pip:

```bash
pip install -r requirements.txt
#Files in this Repository
```
- **real_estates_price_detector.py**: Main Python script for data preprocessing, model training, and evaluation.
- **data.csv**: The dataset used for training the model.
- **price.joblib**: The saved model file after training.

# Steps Involved

### 1. Data Loading and Exploration
- The dataset is loaded using `pandas`, and basic exploratory data analysis (EDA) is performed to understand the dataset better.
- Missing values are handled, and any necessary transformations are applied.

### 2. Data Splitting
- The dataset is split into a training set and a test set (80-20 split).
- Stratified sampling is used to ensure that the distribution of features in both sets is similar.

### 3. Feature Engineering
- New features are created, and correlations between different features and the target (price) are examined.

### 4. Model Training
- A machine learning model (`RandomForestRegressor`) is trained using the training dataset.
- The model is evaluated using Root Mean Squared Error (RMSE) and cross-validation to assess its performance.

### 5. Model Evaluation
- The trained model is tested on the test dataset.
- The performance of the model is evaluated using RMSE.

### 6. Saving the Model
- The trained model is saved using the `joblib` library to make predictions on new data without retraining.

### 7. Making Predictions
- The saved model can be loaded to predict housing prices for new input data.

# How to Use the Model

1. Install the required dependencies using the following command:

    ```bash
    pip install -r requirements.txt
    ```

2. Load the model and make predictions:

    ```python
    from joblib import load
    import numpy as np

    # Load the trained model
    model = load('price.joblib')

    # Example feature set (replace with actual features)
    features = np.array([[-5.43942006, 4.12628155, -1.6165014, -0.67288841, -1.42262747,
                        -11.44443979304, -49.31238772,  7.61111401, -26.0016879 , -0.5778192 ,
                        -0.97491834,  0.41164221, -66.86091034]])

    # Predict the price
    prediction = model.predict(features)
    print(f"Predicted Price: {prediction[0]}")
    ```

# License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
