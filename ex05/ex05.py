"""
Exercise 5: Gradient Descent (Bonus)
Implement gradient descent to find optimal coefficients for Linear Regression.

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
from functools import partial
from multiprocessing import Pool

def compute_mse(coefs, X, y):
    """
    Compute MSE for given coefficients [a, b].
    
    coefs: list [a, b]
    X: features (n_samples, 1)
    y: target (n_samples,)
    """
    a, b = coefs
    # X is (100, 1), we need to flatten for computation
    X_flat = X.reshape(-1)
    y_preds = a * X_flat + b
    mse = np.mean((y_preds - y) ** 2)
    return mse

def gradient_descent(X, y, learning_rate=0.1, n_iterations=100, init_a=0.0, init_b=0.0):
    """
    Implement gradient descent to find optimal a and b.
    
    Returns history of (a, b) through iterations.
    """
    a, b = init_a, init_b
    X_flat = X.reshape(-1)
    n = len(y)
    history = [(a, b)]
    
    for i in range(n_iterations):
        # Predictions
        y_pred = a * X_flat + b
        
        # Gradients
        # MSE = 1/n * sum((a*x + b - y)^2)
        # d(MSE)/da = 2/n * sum((a*x + b - y) * x)
        # d(MSE)/db = 2/n * sum(a*x + b - y)
        grad_a = (2/n) * np.sum((y_pred - y) * X_flat)
        grad_b = (2/n) * np.sum(y_pred - y)
        
        # Update parameters
        a = a - learning_rate * grad_a
        b = b - learning_rate * grad_b
        
        history.append((a, b))
    
    return np.array(history)

def main():
    # Generate dataset (same as exercise 2)
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
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    
    # Plot the data
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, alpha=0.6, label='Data points')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Generated Data for Gradient Descent')
    plt.legend()
    plt.savefig('ex05_data.png')
    plt.show()
    
    # Question 1: Compute MSE for a=1, b=2
    mse_test = compute_mse([1, 2], X, y)
    print(f"\nMSE for a=1, b=2: {mse_test:.4f}")
    
    # Question 2-5: Grid search
    print("\n--- Grid Search ---")
    aa, bb = np.mgrid[-200:200:0.5, -200:200:0.5]
    grid = np.c_[aa.ravel(), bb.ravel()]
    print(f"Grid shape: {grid.shape} (should be 640000, 2)")
    
    # Parallelize MSE computation
    print("Computing MSE for all grid points...")
    with Pool() as pool:
        func = partial(compute_mse, X=X, y=y)
        losses = pool.map(func, grid)
    
    print(f"Computed {len(losses)} MSE values")
    
    # Find optimal a, b from grid
    min_idx = np.argmin(losses)
    optimal_a, optimal_b = grid[min_idx]
    min_mse = losses[min_idx]
    print(f"Optimal from grid: a={optimal_a:.2f}, b={optimal_b:.2f}, MSE={min_mse:.4f}")
    
    # Plot MSE contour
    losses_reshaped = np.array(losses).reshape(aa.shape)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(aa, bb, losses_reshaped, 100, cmap="RdBu", vmin=0, vmax=160000)
    ax_c = fig.colorbar(contour)
    ax_c.set_label("MSE")
    
    ax.set(aspect="equal",
           xlim=(-200, 200),
           ylim=(-200, 200),
           xlabel="$a$",
           ylabel="$b$")
    ax.set_title("MSE Landscape")
    plt.savefig('ex05_mse_contour.png')
    plt.show()
    
    # Plot data with optimal line from grid
    X_flat = X.reshape(-1)
    y_pred_optimal = optimal_a * X_flat + optimal_b
    
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, alpha=0.6, label='Data points')
    plt.plot(X_flat, y_pred_optimal, color='green', linewidth=2, 
             label=f'Grid optimal: y = {optimal_a:.2f}x + {optimal_b:.2f}')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Data with Grid Search Optimal Line')
    plt.legend()
    plt.savefig('ex05_grid_fit.png')
    plt.show()
    
    # Question 8: Gradient Descent
    print("\n--- Gradient Descent ---")
    history = gradient_descent(X, y, learning_rate=0.1, n_iterations=100, init_a=0.0, init_b=0.0)
    print(f"Gradient descent final: a={history[-1, 0]:.4f}, b={history[-1, 1]:.4f}")
    print(f"MSE at final: {compute_mse(history[-1], X, y):.4f}")
    
    # Plot MSE contour with gradient descent path
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(aa, bb, losses_reshaped, 100, cmap="RdBu", vmin=0, vmax=160000)
    ax_c = fig.colorbar(contour)
    ax_c.set_label("MSE")
    
    # Plot gradient descent path
    ax.plot(history[:, 0], history[:, 1], 'w.-', markersize=4, linewidth=1, label='GD path')
    ax.plot(history[0, 0], history[0, 1], 'go', markersize=10, label='Start (0, 0)')
    ax.plot(history[-1, 0], history[-1, 1], 'ro', markersize=10, label='End')
    
    ax.set(aspect="equal",
           xlim=(-200, 200),
           ylim=(-200, 200),
           xlabel="$a$",
           ylabel="$b$")
    ax.set_title("MSE Landscape with Gradient Descent Path")
    ax.legend()
    plt.savefig('ex05_gd_path.png')
    plt.show()
    
    # Compare with Scikit-learn
    print("\n--- Comparison with Scikit-learn ---")
    sk_model = LinearRegression()
    sk_model.fit(X, y)
    print(f"Sklearn: a={sk_model.coef_[0]:.4f}, b={sk_model.intercept_:.4f}")
    print(f"Sklearn MSE: {mean_squared_error(y, sk_model.predict(X)):.4f}")
    print(f"GD final:  a={history[-1, 0]:.4f}, b={history[-1, 1]:.4f}")
    print(f"GD MSE:    {compute_mse(history[-1], X, y):.4f}")

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

if __name__ == "__main__":
    main()