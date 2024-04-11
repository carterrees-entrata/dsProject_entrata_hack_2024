# Logistic Regression Models: Understanding and Application

Logistic regression models are utilized for predicting the probability of a binary outcome based on one or more predictor variables. These models are especially common in fields like medicine, social sciences, and marketing, where the interest is in predicting outcomes that are categorical (e.g., yes/no, success/failure). Unlike linear regression models, which predict a continuous outcome, logistic regression models predict the likelihood of an event occurring.

## What Logistic Regression Models Do:

- **Predict Probabilities:** They estimate the probability of the dependent variable (often represented as 0 or 1) based on the independent variables. This could be predicting whether a patient has a disease (1) or not (0), based on various predictors like age, blood pressure, etc.
- **Binary Outcomes:** Logistic regression is used when the outcome is binary. It provides a framework for modeling the probability that an observation falls into one of two categories.
- **Inform Decision Making:** Understanding the probabilities and the factors that influence them can guide strategic decisions and interventions.

## Example in the Multifamily Housing Industry:

### Predicting Rental Conversion

Imagine a property management company is interested in predicting the probability that a potential renter (lead) will sign a lease (convert) based on their engagement with the company’s website. This insight can help in optimizing marketing strategies and understanding customer behavior better.

#### Independent Variables:
- **Number of Website Visits:** How often the lead visited the website.
- **Number of Virtual Tours:** How many virtual property tours the lead completed.
- **Length of Browsing Session:** The total time spent on the website during a session.

#### Dependent Variable:
- **Rental Conversion:** Whether the lead signed the lease (1) or not (0).

### Model Code Example

```python
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Example dataset
data_df = pd.DataFrame({
    'Number of Website Visits': [10, 2, 5, 8, 15, 4, 9, 1, 3, 7],
    'Number of Virtual Tours': [2, 1, 0, 3, 4, 2, 3, 0, 1, 2],
    'Length of Browsing Session': [30, 5, 10, 20, 45, 12, 25, 3, 8, 18], # in minutes
    'Rental Conversion': [1, 0, 0, 1, 1, 0, 1, 0, 0, 1]
})

# Defining the predictor variables and the target variable
X = data_df[['Number of Website Visits', 'Number of Virtual Tours', 'Length of Browsing Session']]
y = data_df['Rental Conversion']

# Adding a constant for the intercept
X = sm.add_constant(X)

# Fitting the logistic regression model
model = sm.Logit(y, X).fit()

# Printing the model summary
print(model.summary())

# Predicting probabilities
predictions = model.predict(X)
data_df['Predicted Probability'] = predictions

# Display the first few rows of the dataframe to see actual vs predicted probabilities
print(data_df.head())
```

### Logistic Regression Model Interpretation

The logistic regression summary provides a wealth of information about the model's performance and the significance of individual predictors in estimating the probability of a rental lead converting into a signed renter. Below is an interpretation of the pseudo R-squared, coefficients, their significance, and the implications of exponentiating the coefficients:

#### Pseudo R-squared: 0.09342
- **Interpretation:** The Pseudo R-squared value indicates that about 9.34% of the variability in the dependent variable (rental conversion) can be explained by the model. Unlike the R-squared in linear regression, which measures variance explained directly, the pseudo R-squared in logistic regression does not have an absolute interpretation and should be used cautiously. It is primarily used for comparing the goodness-of-fit between different logistic models rather than an absolute measure of fit.
- **Caveat:** It is not equivalent to the R-squared from linear regression; its values are generally lower and do not imply that the model explains 9.34% of all possible variation as in linear regression. It's more of a relative measure of improvement over a model with no predictors.

#### Coefficients and Their Significance
- **Constant (Intercept):** -2.5428
  - **Significance:** The intercept is significant with a p-value of 0.022, indicating that the baseline log-odds of conversion (when all predictors are zero) is statistically significant.
- **Number of Website Visits:** Coefficient = 0.0474
  - **Significance:** With a p-value of 0.621, this predictor is not statistically significant, suggesting that the number of website visits alone does not have a strong predictive power on the likelihood of rental conversion.
- **Number of Virtual Tours:** Coefficient = 0.6224
  - **Significance:** This coefficient is statistically significant (p = 0.002), indicating a strong relationship between the number of virtual tours and the probability of converting a lead into a renter.
  - **Direction:** The positive coefficient implies that as the number of virtual tours increases, so does the probability of rental conversion.
  - **Exponentiated Coefficient (Odds Ratio):** The exponentiated coefficient of 0.6224 is approximately 1.863. This means that each additional virtual tour is associated with an increase in the odds of conversion by 86.3%, holding other factors constant.
- **Length of Browsing Session:** Coefficient = 0.0156
  - **Significance:** This predictor is not significant (p = 0.723), suggesting that the length of browsing session, in minutes, does not significantly impact the likelihood of rental conversion.
  - **Exponentiated Coefficient:** An exponentiated coefficient of 0.0156 translates to a very small odds ratio increase of approximately 1.016. This indicates a 1.6% increase in the odds of conversion for each additional minute spent in a browsing session, although this effect is not statistically significant.

### Conclusion
The logistic regression model reveals that among the predictors examined, the number of virtual tours a potential renter takes is a significant and positive predictor of converting into a signed renter. This finding underscores the importance of engaging potential renters through interactive and virtual experiences. The coefficients for website visits and browsing session length, however, do not show significant effects on rental conversion, suggesting that simply increasing these metrics without enhancing the quality or engagement of the experience may not be effective in increasing conversion rates.

