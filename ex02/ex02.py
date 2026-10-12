"""
Exercise 2: Linear regression in 1D
Generate data, fit model, plot, compute MSE.

Tips:
- Always split your data: Never train and test on the same data; use train_test_split to ensure unbiased model evaluation.
- Understand the math: Linear regression finds the best-fit line by minimizing MSE - knowing this helps debug and improve models.
- Check for linear relationships: Use scatter plots to verify that features have linear relationships with the target before applying linear regression.
- Scale features when needed: Features with different ranges can affect gradient descent convergence - consider standardization for real-world data.
- Start simple: Begin with one feature to understand the model, then add complexity - interpretability is key in linear regression.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

def compute_mse(y_true, y_pred):
    """Compute Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)

def main():
    # Generate dataset with noise=10
    X, y, true_coef = make_regression(
        n_samples=100,
        n_features=1,
        n_informative=1,
        noise=10,
        coef=True,
        random_state=0,
        bias=100.0
    )
    
    print("True coefficient:", true_coef)
    
    # Plot the data
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, alpha=0.6, label='Data points')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Generated Data (noise=10)')
    plt.legend()
    plt.savefig('ex02_data_noise10.png')
    plt.show()
    
    # Fit LinearRegression
    model = LinearRegression()
    model.fit(X, y)
    
    # Equation of fitted line
    coef = model.coef_[0]
    intercept = model.intercept_
    print(f"\nFitted line equation: y = {coef:.4f} * x + {intercept:.4f}")
    
    # Predict on X
    y_pred = model.predict(X)
    
    # Compute MSE
    mse = compute_mse(y, y_pred)
    print(f"MSE (noise=10): {mse:.4f}")
    
    # Add fitted line to plot
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, alpha=0.6, label='Data points')
    plt.plot(X, y_pred, color='red', linewidth=2, label=f'Fitted line: y = {coef:.2f}x + {intercept:.2f}')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Linear Regression Fit (noise=10)')
    plt.legend()
    plt.savefig('ex02_fit_noise10.png')
    plt.show()
    
    # Now change noise to 50 and repeat
    print("\n--- With noise=50 ---")
    X2, y2, true_coef2 = make_regression(
        n_samples=100,
        n_features=1,
        n_informative=1,
        noise=50,
        coef=True,
        random_state=0,
        bias=100.0
    )
    
    print("True coefficient:", true_coef2)
    
    # Fit LinearRegression
    model2 = LinearRegression()
    model2.fit(X2, y2)
    
    coef2 = model2.coef_[0]
    intercept2 = model2.intercept_
    print(f"Fitted line equation: y = {coef2:.4f} * x + {intercept2:.4f}")
    
    # Predict and compute MSE
    y_pred2 = model2.predict(X2)
    mse2 = compute_mse(y2, y_pred2)
    print(f"MSE (noise=50): {mse2:.4f}")
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(X2, y2, alpha=0.6, label='Data points')
    plt.plot(X2, y_pred2, color='red', linewidth=2, label=f'Fitted line: y = {coef2:.2f}x + {intercept2:.2f}')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Linear Regression Fit (noise=50)')
    plt.legend()
    plt.savefig('ex02_fit_noise50.png')
    plt.show()

if __name__ == "__main__":
    main()