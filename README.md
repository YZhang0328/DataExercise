### Two Python scripts designed to analyze data from (X, Y, Z). 
### 1. DataVisualization.py
Loads and visualizes the data.
### 2. LinearRegression_with_QR.py
This script builds linear regression models using the datasets and evaluates their performance.
### Features: 
### Data Preprocessing<br>
### Basic Linear Regression: 
1. Builds a simple linear regression model using X features;<br>
2. Evaluates the model with Mean Squared Error (MSE) and R-squared metrics.<br>
### Augmented Linear Regression: <br>
1. Adds interaction terms between X and Z features using QR decomposition.<br>
2. Selects significant interaction terms based on residual minimization.<br>
3. Fits a new linear regression model with augmented features.<br>
4. Evaluates the model with Mean Squared Error (MSE) and R-squared metrics.<br>

### How to Run:

To run the data visualization:
```python
python DataExercise\DataVisualization.py
```

To run the regression model:
```python
python DataExercise\linearRegression_with_QR_decomposition.py
