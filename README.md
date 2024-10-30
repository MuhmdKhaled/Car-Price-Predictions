Car Price Prediction
-------------------------------------------------------------------------------------------------------------------------------------------------------------
This project builds a machine learning model to predict car prices based on various attributes, using Python libraries for data analysis and visualization.

Features
----------------------------------------------------------------------------------------------------------------------------------
The code includes the following functionalities:

Data Loading: Loads a dataset (CSV format) into a DataFrame using pandas.

Data Exploration: Displays basic information on the dataset's shape, columns, and top values for each column, alongside the unique count of entries per column.

Data Cleaning: Renames columns to lower case and removes spaces, preparing the data for modeling. Non-numeric columns are standardized to lower case.

Visualization: (Seaborn, Matplotlib, Plotly) – Includes tools for visualizing data distributions and relationships.

Encoding: Uses LabelEncoder for categorical variables.

Data Splitting and Scaling: Splits data into training and test sets and scales features for model compatibility.

Model Training: Implements a RandomForestRegressor for predicting car prices, with performance metrics such as R2 Score and Mean Squared Error.

-Evaluation: Calculates model performance using metrics to assess prediction accuracy.

Overview
-------------------------------------------------------------------------------------------------------------------------------------
The project uses a dataset of car features and prices to create a predictive model with the following main steps:

1. Libraries and Setup
--------------------------------------------------------------------------------------------------------------------------------------------------------------
Imports essential libraries for data manipulation (pandas, numpy), visualization (seaborn, matplotlib, plotly), and machine learning (scikit-learn).
Sets up in-line plotting for Jupyter notebooks.

2. Data Loading and Initial Exploration
----------------------------------------------------------------------------------------------------------------------------------------------------------------
Loads the dataset into a DataFrame and outputs basic structure:
Shape and column names for an overview of data dimensions and features.
Data types and missing values summary to understand preprocessing needs.
Displays unique value counts for each column to highlight variability in categorical and numeric columns.

3. Data Cleaning
-------------------------------------------------------------------------------------------------------------------------------------------------
Column renaming: Standardizes column names to lower case with underscores for easier access.
String formatting: Converts string columns to lower case and removes unnecessary characters to prevent inconsistencies.

4. Encoding Categorical Data
---------------------------------------------------------------------------------------------------------------------------------------------------------
Applies LabelEncoder to categorical columns, transforming them into numeric representations compatible with machine learning algorithms.

5. Feature Scaling and Data Splitting
------------------------------------------------------------------------------------------------------------------------------------------------------
Splits the data into training and testing subsets to evaluate model performance objectively.
Uses StandardScaler to normalize feature values, improving model stability.

6. Model Selection and Training
------------------------------------------------------------------------------------------------------------------------
Builds a pipeline with RandomForestRegressor, a versatile and robust ensemble method for regression tasks.
Trains the model on the processed data to learn the relationship between car attributes and prices.

7. Model Evaluation
-------------------------------------------------------------------------------------------------------------------------------------------------
Measures model performance using R² Score and Mean Squared Error (MSE), providing insights into the model’s accuracy and error margins.

Trains the model on the processed data to learn the relationship between car attributes and prices.
9. Model Evaluation
Measures model performance using R² Score and Mean Squared Error (MSE), providing insights into the model’s accuracy and error margins.
