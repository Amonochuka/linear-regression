"""
Exercise 1: Scikit-learn estimator
Fit a LinearRegression model and predict.

Tips:
- Always split your data: Never train and test on the same data; use train_test_split to ensure unbiased model evaluation.
- Understand the math: Linear regression finds the best-fit line by minimizing MSE - knowing this helps debug and improve models.
- Check for linear relationships: Use scatter plots to verify that features have linear relationships with the target before applying linear regression.
- Scale features when needed: Features with different ranges can affect gradient descent convergence - consider standardization for real-world data.
- Start simple: Begin with one feature to understand the model, then add complexity - interpretability is key in linear regression.
"""

import numpy as np
from sklearn.linear_model import LinearRegression

def main():
    # Given data
    X = np.array([[1], [2.1], [3]])
    y = np.array([[1], [2], [3]])
    x_pred = np.array([[4]])
    
    # Fit LinearRegression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict for x_pred = [[4]]
    prediction = model.predict(x_pred)
    
    # Print coefficients, intercept, and score
    print("Coefficients (coef_):", model.coef_)
    print("Intercept (intercept_):", model.intercept_)
    print("Score (R^2):", model.score(X, y))
    print("Prediction for [[4]]:", prediction)

if __name__ == "__main__":
    main()