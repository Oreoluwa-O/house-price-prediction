# House Price Regression (Ames Housing)

This project predicts house sale prices using the Ames Housing dataset. It trains and compares a linear regression model and a random forest model in Python using scikit-learn.

## Dataset

- Ames Housing dataset (public dataset from Kaggle).
- Target variable: `SalePrice`.

## Methods

- Linear Regression (baseline).
- Random Forest Regressor (tree-based model).

## Evaluation

Models are evaluated using:

- Root Mean Squared Error (RMSE).
- R-squared (R²).

## How to Run (Locally)

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Put the dataset file in the `data` folder.

3. Run the main script:
   ```bash
   python house_price_regression.py
   ```

The script prints RMSE and R² for each model.
