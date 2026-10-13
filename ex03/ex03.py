"""
Exercise 3: Train test split
Split dataset using train_test_split with shuffle=False, test_size=20%.

Tips:
- Always split your data: Never train and test on the same data; use train_test_split to ensure unbiased model evaluation.
- Understand the math: Linear regression finds the best-fit line by minimizing MSE - knowing this helps debug and improve models.
- Check for linear relationships: Use scatter plots to verify that features have linear relationships with the target before applying linear regression.
- Scale features when needed: Features with different ranges can affect gradient descent convergence - consider standardization for real-world data.
- Start simple: Begin with one feature to understand the model, then add complexity - interpretability is key in linear regression.
"""

import numpy as np
from sklearn.model_selection import train_test_split

def main():
    # Given data
    X = np.arange(1, 21).reshape(10, -1)
    y = np.arange(1, 11)
    
    print("Original X shape:", X.shape)
    print("Original y shape:", y.shape)
    print("X:\n", X)
    print("y:", y)
    
    # Split with shuffle=False, test_size=0.2 (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False, random_state=42
    )
    
    print("\n--- Train/Test Split (shuffle=False, test_size=20%) ---")
    print("X_train shape:", X_train.shape)
    print("X_train:\n", X_train)
    print("y_train shape:", y_train.shape)
    print("y_train:", y_train)
    print("X_test shape:", X_test.shape)
    print("X_test:\n", X_test)
    print("y_test shape:", y_test.shape)
    print("y_test:", y_test)

if __name__ == "__main__":
    main()