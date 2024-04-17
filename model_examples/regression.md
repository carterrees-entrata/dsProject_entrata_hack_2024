# Regression Models: What They Do and How to Use Them

Regression models are a cornerstone of statistical analysis and predictive modeling, widely used across industries for forecasting, understanding relationships between variables, and decision making. At their core, regression models analyze the relationship between a dependent (or target) variable and one or more independent (or predictor) variables. The goal is to understand how changes in the predictors influence the outcome and to predict future values of the dependent variable based on known values of the predictors. Note: the dependent variable in this type of regression model is continuous and not a categorical measure.

## What Regression Models Do:

- **Predict Outcomes:** They estimate the value of the dependent variable based on the independent variables. For instance, predicting a house's price (dependent variable) based on its size, location, and age (independent variables).
- **Quantify Relationships:** Regression models help in quantifying the strength and direction of relationships between variables. This can aid in identifying which factors are most influential.
- **Inform Decision Making:** By understanding how variables are related and predicting outcomes, organizations can make informed decisions.

## Example in the Multifamily Housing Industry:

### Predicting Apartment Damage Deposit Use

Let’s imagine a multifamily housing corporation wants to predict the amount of an apartment’s damage deposit that will be used by the end of a lease. This prediction helps in financial planning, risk assessment, and identifying patterns that could guide future leasing policies.

#### Independent Variables:
- **Rental Agreement Length:** The initial duration of the lease (e.g., 12 months).
- **Number of Times a Lease Was Renewed:** Indicates tenant stability and longer occupancy periods.
- **Number of Children:** Children in a household could correlate with wear and tear on the property.

#### Dependent Variable:
- **Damage Deposit Used:** The portion of the damage deposit that was utilized to cover repairs and cleaning after the tenants moved out.

### Model Code Example


```python
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Create a DataFrame, this is just for example purposes. You will want at least 100 cases for your project. 

data_df = pd.DataFrame({
    'Rental Agreement Length (months)': [12, 24, 12, 36, 12, 24, 12, 24, 36, 12],
    'Number of Times Lease Renewed': [0, 1, 0, 2, 1, 2, 0, 3, 1, 2],
    'Number of Children': [2, 0, 1, 0, 3, 2, 0, 1, 4, 2],
    'Damage Deposit Used ($)': [200, 150, 180, 160, 220, 200, 100, 190, 250, 210]
})

### Separating features (X) and target variable (y)
X = data_df[['Rental Agreement Length (months)', 'Number of Times Lease Renewed', 'Number of Children']]
y = data_df['Damage Deposit Used ($)']

### Adding a constant to the model (intercept)
X = sm.add_constant(X)

### Fitting the model
model = sm.OLS(y, X).fit()

### Predicting the values
y_pred = model.predict(X)

### Plotting Actual vs Predicted values
plt.scatter(y, y_pred)
plt.xlabel('Actual Damage Deposit Used ($)')
plt.ylabel('Predicted Damage Deposit Used ($)')
plt.title('Actual vs Predicted Damage Deposit Used')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
plt.show()

### Printing the summary table
print(model.summary())

```
### Regression Analysis Interpretation

The regression table provides insights into how well the model explains the variation in the dependent variable, `Damage Deposit Used ($)`, and the significance of each independent variable. Here's a detailed interpretation:

### Model Fit

- **R-squared:** 0.885
  - This indicates that approximately 88.5% of the variability in the damage deposit used can be explained by the model's independent variables, which suggests a strong fit.

### Coefficients and Significance

- **Intercept (`const`):** 130.0720
  - Interpretation: When all the independent variables are zero, the damage deposit used is expected to be $130.07. This coefficient is statistically significant (p < 0.001).

- **Rental Agreement Length (months):** 0.1287
  - Interpretation: For each additional month in the rental agreement length, the damage deposit used increases by approximately $0.13, holding other variables constant. This effect is not statistically significant (p = 0.853), indicating a weak or nonexistent impact on the damage deposit used.

- **Number of Times Lease Renewed:** 9.6293
  - Interpretation: Each additional lease renewal is associated with an increase of about $9.63 in the damage deposit used, holding other variables constant. The p-value of 0.180 suggests that this relationship is not statistically significant at the 0.05 level, although it approaches significance and might indicate a trend worth further investigation.

- **Number of Children:** 27.8320
  - Interpretation: The presence of each additional child in the household is associated with an increase of about $27.83 in the damage deposit used, holding other variables constant. This relationship is statistically significant (p = 0.001), highlighting the importance of the number of children as a predictor.

### Conclusion

The high R-squared value indicates a good overall fit of the model to the observed data. However, careful consideration is required when interpreting the impact of individual predictors, especially given the non-significance of the rental agreement length and the borderline significance of the number of times a lease is renewed. The number of children emerges as a clear, significant predictor of the damage deposit used.
