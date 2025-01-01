import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


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


def preprocess_and_standardize(X, Z) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Preprocess and standardize the data."""

    def cap_outliers(column):
        lower_bound = np.percentile(column, 0.01)
        upper_bound = np.percentile(column, 99.99)
        return np.clip(column, lower_bound, upper_bound)

    # Replace missing values and cap outliers
    X_cleaned = X.apply(lambda col: col.fillna(col.median())).apply(cap_outliers, axis=0)
    Z_cleaned = Z.apply(lambda col: col.fillna(col.median())).apply(cap_outliers, axis=0)

    # Standardizes the features of a dataset by removing the mean and scaling to unit variance.
    scaler_X = StandardScaler()
    scaler_Z = StandardScaler()

    X_standardized = pd.DataFrame(scaler_X.fit_transform(X_cleaned), columns=X_cleaned.columns)
    Z_standardized = pd.DataFrame(scaler_Z.fit_transform(Z_cleaned), columns=Z_cleaned.columns)

    return X_standardized, Z_standardized


def model_with_X(X, Y) -> tuple[float, float, pd.Series, np.ndarray]:
    """Build a linear regression model using only X as the input features."""

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=50)

    # Fit the model
    model = LinearRegression()
    model.fit(X_train, Y_train)

    # Make predictions
    Y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)

    print("Model with X only:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R-squared: {r2:.4f}")

    return mse, r2, Y_test, Y_pred


def add_interactions_with_qr(X_train_values, Z_train, residual, tolerance=1) -> tuple[np.ndarray, list]:
    """
    Add interaction terms based on QR decomposition and residual minimization according to the proposed algorithm in Theory B.
    Parameters: the splited training data X, Z, and the residual of the linear regression model trained on X.
    Returns: the augmented X matrix and the list of selected terms.
    """
    
    # Perform QR decomposition of the training data in X1
    Q, R = np.linalg.qr(X_train_values, mode='reduced')
    selected_terms = []
    updated_X = X_train_values
    updated_residual = residual.values
    # Generate all possible interaction terms between X and Z
    interaction_candidates = [
        (x_col, z_col) for x_col in range(X_train_values.shape[1]) for z_col in Z_train.columns
    ]

    # Store already used interactions to avoid recomputation
    applied_interactions = set()

    for x_col, z_col in interaction_candidates:
        if (x_col, z_col) in applied_interactions:
            continue  # Skip if interaction has already been added

        # Construct the interaction term
        interaction = X_train_values[:, x_col] * Z_train[z_col].values

        # Use QR decomposition to calculate residuals of X Z interaction terms; the theory of the computation method is given by the Theory B solution;
        # Using the residual instead of the original interaction term is to make sure Z only contributes the information that not ready in X.
        ej = interaction.reshape(-1, 1) - Q @ (Q.T @ interaction.reshape(-1, 1))
        # Compute the slope coefficient gamma_j and residual reduction
        gamma_j = (ej.T @ updated_residual) / (ej.T @ ej)
        # Update the residual after adding the interaction residuals
        new_residual = (updated_residual - gamma_j * ej).T @ (updated_residual - gamma_j * ej)

        # Check if the new residual is smaller than the original one, if so, add the interaction residual term to the model
        if new_residual < updated_residual.T @ updated_residual - tolerance:
            updated_X = np.hstack([updated_X, ej])
            updated_residual = updated_residual - (ej.T @ updated_residual) / (ej.T @ ej) * ej
            selected_terms.append((x_col, z_col))
            applied_interactions.add((x_col, z_col))
            print(f"Added interaction term: ({x_col}, {z_col}), New residual: {float(new_residual):.4f}")

    return updated_X, selected_terms


def model_with_X_and_interactions(X, Z, Y) -> tuple[float, float, pd.Series, np.ndarray]:
    """Build a forecasting model using X and multiple interaction terms with Z."""

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=50)
    Z_train, Z_test = train_test_split(Z, test_size=0.1, random_state=50)

    # Initial model with X
    model = LinearRegression()
    model.fit(X_train, Y_train)

    # Linear regression residuals
    residual = Y_train.values - X_train @ model.coef_.T - model.intercept_

    # Add multiple interaction terms
    X_augmented, selected_terms = add_interactions_with_qr(X_train.values, Z_train, residual)

    # Fit the augmented model
    model_augmented = LinearRegression()
    model_augmented.fit(X_augmented, Y_train)
    Q_test, R = np.linalg.qr(X_test, mode='reduced')

    # Augment test data using the interaction residual terms
    X_test_augmented = X_test.values
    for x_col, z_col in selected_terms:
        interaction = X_test.values[:, x_col] * Z_test[z_col].values
        ej = interaction.reshape(-1, 1) - Q_test @ (Q_test.T @ interaction.reshape(-1, 1))
        X_test_augmented = np.hstack([X_test_augmented, ej])

    # Make predictions
    Y_pred = model_augmented.predict(X_test_augmented)

    # Evaluate the model
    mse = mean_squared_error(Y_test.values, Y_pred)
    r2 = r2_score(Y_test.values, Y_pred)

    print("Model with X and multiple interaction terms:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R-squared: {r2:.4f}")

    return mse, r2, Y_test, Y_pred


def plot_real_vs_predicted(Y_test, Y_pred, title) -> None:
    """Plot real vs. predicted values for comparison."""

    plt.figure(figsize=(10, 6))
    plt.plot(Y_test.values, label="Real Values", linestyle="--", marker="o", alpha=0.7)
    plt.plot(Y_pred, label="Predicted Values", linestyle="-", marker="x", alpha=0.7)
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Values")
    plt.legend()
    plt.grid(True)
    plt.show()


def main() -> None:
    """Main function to build models and evaluate performance."""

    X, Y, Z = load_data()
    X_standardized, Z_standardized = preprocess_and_standardize(X, Z)

    # Model with X only
    mse_x, r2_x, Y_test, Y_pred_x = model_with_X(X_standardized, Y)

    # Plot real vs predicted for model with X only
    plot_real_vs_predicted(Y_test, Y_pred_x, "Real vs Predicted: Model with X Only")
   
    # Model with X and multiple interaction terms
    mse_xz, r2_xz, Y_test_xz, Y_pred_xz = model_with_X_and_interactions(X_standardized, Z_standardized, Y)

    # Plot real vs predicted for model with X and interaction terms
    plot_real_vs_predicted(Y_test_xz, Y_pred_xz, "Real vs Predicted: Model with X and Interaction Terms")

    # Compare models
    print("\nComparison:")
    print(f"Model with X only: MSE = {mse_x:.4f}, R-squared = {r2_x:.4f}")
    print(f"Model with X and multiple interaction terms: MSE = {mse_xz:.4f}, R-squared = {r2_xz:.4f}")


if __name__ == "__main__":
    main()
