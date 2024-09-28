# Libraries
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import HelperClass as hc 
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from IPython.display import display, HTML


class HelperClass:

    def __init__(self):
        pass

    def DataSplitter(self, data, independent_vars, dependent_var, testSize = 0.2):
        # Select features (independent variables) and target (dependent variable)
        X = data[independent_vars]  # Independent variables
        y = data[dependent_var]  # Dependent variable

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=42)

        return X_train, X_test, y_train, y_test
    
    def CorrelationMatrixHeatMap(self, data):
        # Calculate the correlation matrix for numeric columns in the dataset
        correlation_matrix = data.corr()

        # Display the correlation matrix
        correlation_matrix

        # Set up the matplotlib figure
        plt.figure(figsize=(8, 6))

        # Generate a heatmap for the correlation matrix
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)

        # Show the plot
        plt.title('Correlation Matrix Heatmap')
        plt.show()

    def LinearRegressionFitter(self, X_train, y_train, X_test, y_test):
        # Add intercept manually for X_train and X_test
        X_train_with_intercept = np.column_stack((np.ones(X_train.shape[0]), X_train))
        X_test_with_intercept = np.column_stack((np.ones(X_test.shape[0]), X_test))

        # Create a Linear Regression model
        model = LinearRegression()

        # Fit the model to the training data
        model.fit(X_train, y_train)

        # Predict on the train data
        y_train_pred = model.predict(X_train)

        # Predict on the test data
        y_pred = model.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return {"model": model, "mse": mse, "r2": r2}

    def ModelSummaryStats(self, my_model, mse, r2, header, independent_vars):
        regression_str = 'Insurance Premium = '
        for i in range(len(independent_vars)):
            coeffient = my_model.coef_[0][i]
            var_name = independent_vars[i]

            if(i == 0):
                regression_str = regression_str + f" {coeffient:.2f}({var_name})"
            else:
                regression_str = regression_str + f" + {coeffient:.2f}({var_name})"

        regression_str = regression_str + f" + {my_model.intercept_[0]:.2f}"
        
        # Create an HTML string
        html_model_output = f"""
        <h1>{header}</h1>
        <h2>Linear Regression Model Summary</h2>
        <p><b>Mean Squared Error:</b> {{:.2f}}</p>
        <p><b>R-squared:</b> {{:.4f}}</p>
        <p><b>Regression Equation:</b></p>
        <p>{regression_str}</p>
        """.format(mse, r2, my_model.coef_[0][0], my_model.coef_[0][1], my_model.coef_[0][2], my_model.intercept_[0])

        # Display the formatted output
        display(HTML(html_model_output))