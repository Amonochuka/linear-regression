"""
Exercise 4: Forecast diabetes progression
Use Linear Regression to forecast diabetes progression.

Tips:
- Always split your data: Never train and test on the same data; use train_test_split to ensure unbiased model evaluation.
- Understand the math: Linear regression finds the best-fit line by minimizing MSE - knowing this helps debug and improve models.
- Check for linear relationships: Use scatter plots to verify that features have linear relationships with the target before applying linear regression.
- Scale features when needed: Features with different ranges can affect gradient descent convergence - consider standardization for real-world data.
- Start simple: Begin with one feature to understand the model, then add complexity - interpretability is key in linear regression.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def main():
    # Load diabetes dataset
    diabetes = load_diabetes(as_frame=True)
    X, y = diabetes.data, diabetes.target
    
    print("Dataset shape:", X.shape)
    print("Features:", X.columns.tolist())
    print("Target name: disease progression (quantitative measure)")
    print("\nFirst 5 rows of X:")
    print(X.head())
    print("\nTarget stats:")
    print(y.describe())
    
    # Split dataset: 80% train, 20% test, random_state=43
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=43
    )
    
    print(f"\nTrain set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Fit Linear Regression on all variables
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Coefficients and intercept
    print("\n--- Linear Regression Results ---")
    print("Intercept:", model.intercept_)
    print("Coefficients:")
    for feature, coef in zip(X.columns, model.coef_):
        print(f"  {feature}: {coef:.4f}")
    
    # Equation
    equation_parts = [f"{model.intercept_:.4f}"]
    for feature, coef in zip(X.columns, model.coef_):
        sign = "+" if coef >= 0 else ""
        equation_parts.append(f"{sign}{coef:.4f}*{feature}")
    equation = "y = " + " ".join(equation_parts)
    print(f"\nEquation: {equation}")
    
    # Predict on test set
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    
    # Compute MSE on train and test sets
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    
    print(f"\nMSE on train set: {mse_train:.4f}")
    print(f"MSE on test set: {mse_test:.4f}")
    
    # Show some predictions vs actual
    print("\n--- Sample Predictions (Test Set) ---")
    for i in range(min(5, len(y_test))):
        print(f"  Actual: {y_test.iloc[i]:.2f}, Predicted: {y_pred_test[i]:.2f}")

if __name__ == "__main__":
    main()