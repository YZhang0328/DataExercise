import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the X, Y, Z data."""

    def load_csv(file_name: str) -> pd.DataFrame:
        """Helper function to load a CSV file and set the index name."""
        
        file_path = os.path.join(os.getcwd(), file_name)
        df = pd.read_csv(file_path, index_col=[0], header=[0])
        df.index.name = "Index"
        return df

    # Load datasets using the helper function
    X = load_csv("X.csv")
    Y = load_csv("Y.csv")
    Z = load_csv("Z.csv")

    return X, Y, Z

def preprocess_and_clean(X, Z) -> pd.DataFrame:
    """Preprocess and clean the data."""

    def cap_outliers(column):
        lower_bound = np.percentile(column, 0.01)
        upper_bound = np.percentile(column, 99.99)
        return np.clip(column, lower_bound, upper_bound)

    # Replace missing values and cap outliers in X
    X_cleaned = X.apply(lambda col: col.fillna(col.median())).apply(cap_outliers, axis=0)
    return X_cleaned

def visualize_time_series(X_cleaned, Y, Z):
    """Visualize the layers of X_cleaned, Y, and Z as line plots."""

    plt.figure(figsize=(15, 10))

    # Plot X_cleaned time series
    plt.subplot(3, 1, 1)
    for col in X_cleaned.columns:
        plt.plot(X_cleaned.index, X_cleaned[col], label=f"X: {col}", alpha=0.6)
    plt.title("Time Series of Cleaned X Features")
    plt.xlabel("Index")
    plt.ylabel("X Values")
    plt.legend(loc="upper right")

    # Plot Y time series
    plt.subplot(3, 1, 2)
    plt.plot(Y.index, Y.values.flatten(), label="Y", color="red", alpha=0.7)
    plt.title("Time Series of Y")
    plt.xlabel("Index")
    plt.ylabel("Y Values")
    plt.legend(loc="upper right")

    # Plot Z time series
    plt.subplot(3, 1, 3)
    for col in Z.columns:
        plt.plot(Z.index, Z[col], label=f"Z: {col}", alpha=0.6)
    plt.title("Time Series of Z Features")
    plt.xlabel("Index")
    plt.ylabel("Z Values")
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.show()

def main():
    """Main function to visualize the data."""

    X, Y, Z = load_data()
    X_cleaned = preprocess_and_clean(X, Z)

    # Visualize data layers as time series
    visualize_time_series(X_cleaned, Y, Z)

if __name__ == "__main__":
    main()