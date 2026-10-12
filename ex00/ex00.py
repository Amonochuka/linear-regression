"""
Exercise 0: Environment and libraries
This script verifies that all required libraries are installed and working.

Tips:
- Always split your data: Never train and test on the same data; use train_test_split to ensure unbiased model evaluation.
- Understand the math: Linear regression finds the best-fit line by minimizing MSE - knowing this helps debug and improve models.
- Check for linear relationships: Use scatter plots to verify that features have linear relationships with the target before applying linear regression.
- Scale features when needed: Features with different ranges can affect gradient descent convergence - consider standardization for real-world data.
- Start simple: Begin with one feature to understand the model, then add complexity - interpretability is key in linear regression.
"""

import sys
import pandas as pd
import numpy as np
import matplotlib
import sklearn

def check_environment():
    """Check that all required libraries are installed with correct versions."""
    print("Python version:", sys.version)
    print("pandas version:", pd.__version__)
    print("numpy version:", np.__version__)
    print("matplotlib version:", matplotlib.__version__)
    print("scikit-learn version:", sklearn.__version__)
    
    # Verify we can import key modules
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_regression, load_diabetes
    from sklearn.metrics import mean_squared_error
    
    print("\nAll imports successful!")
    print("Environment is ready for linear regression exercises.")

if __name__ == "__main__":
    check_environment()