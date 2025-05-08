# Car Price Prediction Using Machine Learning

## Overview

This repository showcases a comprehensive machine learning project focused on predicting the prices of used cars based on various features such as make, model, year, fuel type, transmission type, and more. By leveraging historical data from CarDekho, this project aims to create an accurate and user-friendly tool that enhances customer experience and streamlines the pricing process for sales representatives. The final product is a deployed Streamlit application that allows users to input car details and receive instant price predictions.

## Project Structure

### 1. **Jupyter Notebook**
   - **File**: `car_dekho_cleaning_and_modeling.ipynb`
   - **Description**: This notebook documents the entire process of data cleaning, feature engineering, model selection, training, and evaluation. The notebook includes exploratory data analysis (EDA) to understand the key features influencing car prices, followed by multiple machine learning models, such as Linear Regression, Decision Trees, Random Forest, and Gradient Boosting, with hyperparameter tuning for optimal performance.

### 2. **Streamlit Application**
   - **File**: `car_dekho_app.py`
   - **Description**: The Streamlit application provides an interactive interface where users can input car specifications and obtain a predicted price. The app is designed to be intuitive and easy to use, making it accessible for both customers and sales representatives.

### 3. **Project Report**
   - **File**: `project_report.pdf`
   - **Description**: A detailed report that covers the entire project lifecycle, from problem statement and data preprocessing to model evaluation and deployment. It includes justifications for the chosen methodologies, a summary of results, and insights derived from the analysis.

### 4. **User Guide**
   - **File**: `user_guide.pdf`
   - **Description**: A step-by-step guide that explains how to use the Streamlit application effectively. It provides instructions on navigating the app, inputting data, and interpreting the results, making it easy for users of all technical backgrounds to interact with the model.

### 5. **Resources**
   - **File**: `resources.zip`
   - **Contents**: This ZIP file contains all necessary resources, including the cleaned dataset used for training, joblib files for saving and loading models, and any additional resources that support the project.

## Getting Started

### Prerequisites
To run the notebook and Streamlit application, ensure you have the following installed:

- **Python 3.7+**
- Required Python libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `streamlit`

 Using the Streamlit Application
- Input Fields: The application allows users to input various car attributes such as make, model, year, fuel type, transmission type, mileage, and more.
- Price Prediction: After entering the details, click the 'Predict' button to get the estimated price of the car.
- User-Friendly Interface: The interface is designed to be intuitive, making it easy for both tech-savvy and non-technical users to interact with the model.

Model Training and Evaluation
- Data Cleaning: The dataset undergoes rigorous cleaning to handle missing values, encode categorical features, and scale numerical features appropriately.
- Model Selection: Several models were trained and evaluated, including Linear Regression, Decision Trees, Random Forest, and Gradient Boosting. Hyperparameter tuning was performed to optimize model performance.
- Evaluation Metrics: The models were evaluated based on Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R²) values. These metrics helped in comparing model performance and selecting the best model for deployment.

Results
- Best Model: The model with the highest R² score and lowest MSE and MAE was selected for the final deployment.
- Accuracy: The deployed model demonstrated strong predictive accuracy, making it reliable for practical use in estimating car prices.

Repository Structure
- car_dekho_cleaning_and_modeling.ipynb: The notebook containing data processing, model training, and evaluation.
- car_dekho_app.py: The Streamlit application script.
- project_report.pdf: The comprehensive project report.
- user_guide.pdf: A detailed user guide for the Streamlit application.
- Car_Price_Prediction.zip: A ZIP file containing the cleaned dataset, joblib files, and other supporting resources.

Acknowledgements
- This project was built using data from CarDekho and relies on several open-source libraries, including Scikit-learn for machine learning and Streamlit for web application deployment.

License
- This project is licensed under the MIT License.

References
- CarDekho
- Scikit-learn Documentation
- Streamlit Documentation
