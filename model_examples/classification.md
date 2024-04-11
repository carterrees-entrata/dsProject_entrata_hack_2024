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
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Create synthetic data
np.random.seed(42)
data_size = 100  # Increased dataset size

# Generate random data
data = {
    'Number of Website Visits': np.random.poisson(5, data_size),
    'Number of Virtual Tours': np.random.poisson(2, data_size),
    'Length of Browsing Session': np.random.normal(20, 5, data_size),  # Average 20 minutes, std dev 5
    'Rental Conversion': np.random.binomial(1, 0.5, data_size)  # Random binary outcome
}

df = pd.DataFrame(data)

# Given this is simulated data for example purposes, we need to introduce more structure to make sure 'Number of Virtual Tours' significant
# Increasing the probability of conversion based on the number of virtual tours
df['Rental Conversion'] = df.apply(
    lambda row: np.random.binomial(1, min(1, 0.1 + 0.15 * row['Number of Virtual Tours'])), axis=1
)

# Define predictors and outcome
X = df[['Number of Website Visits', 'Number of Virtual Tours', 'Length of Browsing Session']]
y = df['Rental Conversion']

# Splitting data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Add a constant for the logistic regression intercept
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

# Fit the logistic regression model
model = sm.Logit(y_train, X_train).fit()

# Print the summary of the logistic regression model
print(model.summary())

# Predicting probabilities for test set
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# Calculate accuracy, precision, recall, and F1 score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the metrics
print(f"Model Accuracy: {accuracy:.2f}")
print(f"Model Precision: {precision:.2f}")
print(f"Model Recall: {recall:.2f}")
print(f"Model F1 Score: {f1:.2f}")

```

### Logistic Regression Model Interpretation

The logistic regression summary table offers detailed insights into the model's effectiveness at predicting whether a rental lead will convert into a signed renter:

#### Pseudo R-squared: 0.08912
- **Interpretation:** This value suggests that approximately 8.91% of the variability in the dependent variable (rental conversion) is accounted for by the model. Unlike the R-squared in linear regression, pseudo R-squared in logistic regression provides a relative measure of model fit rather than an absolute one, useful for comparing models but not as a definitive measure of quality.
- **Caveat:** Pseudo R-squared values in logistic regression are inherently lower and less intuitive than those in linear regression. They do not suggest that the model explains a straightforward percentage of total variance as in other types of regression models.

#### Coefficients and Their Significance
- **Constant (Intercept):** -2.3203
  - **Significance:** The intercept's p-value is 0.055, which is close to the typical significance level of 0.05, suggesting that it is marginally significant. This implies that the baseline log-odds of conversion, when all predictors are zero, is significantly different from zero at the 10% level but just misses being significant at the 5% level.
  
- **Number of Website Visits:** Coefficient = 0.0067
  - **Significance:** With a p-value of 0.954, this predictor is not statistically significant, indicating that the number of website visits does not meaningfully predict the likelihood of rental conversion. This suggests that simply increasing website visits might not be effective in enhancing conversion rates.
  
- **Number of Virtual Tours:** Coefficient = 0.6122
  - **Significance:** This coefficient is statistically significant (p = 0.006), showing a strong positive relationship between the number of virtual tours and the likelihood of converting a lead into a renter.
  - **Direction:** The positive coefficient indicates that as the number of virtual tours increases, so does the probability of rental conversion.
  - **Exponentiated Coefficient (Odds Ratio):** The exponentiated coefficient for this predictor is approximately e^0.6122 ≈ 1.844. This means that each additional virtual tour increases the odds of conversion by about 84.4%, holding other factors constant.
  
- **Length of Browsing Session:** Coefficient = 0.0153
  - **Significance:** With a p-value of 0.755, this predictor is not statistically significant, suggesting that the length of browsing session does not significantly impact the likelihood of rental conversion. It shows that while longer sessions might intuitively seem beneficial, they don’t necessarily translate to higher conversion chances in this model.

Here's an interpretation of the confusion matrix metrics provided for your logistic regression model:

### Model Performance Metrics Interpretation

- **Model Accuracy (0.75 or 75%)**: This metric tells you the overall percentage of correct predictions made by the model, including both true positives and true negatives. An accuracy of 75% suggests that, overall, three out of every four predictions made by the model are correct. While this seems fairly high, accuracy alone might not give a complete picture, especially if the dataset is imbalanced (i.e., if there are significantly more examples of one class than another).

- **Model Precision (1.00 or 100%)**: Precision measures the accuracy of the positive predictions. A precision of 100% means that every prediction made by the model where the lead was predicted to rent (positive class) was correct. There were no false positives, i.e., no cases where the model incorrectly predicted that a lead would rent when they did not. This suggests that the model is very reliable when it predicts a conversion will occur.

- **Model Recall (0.29 or 29%)**: Recall, or sensitivity, measures the model’s ability to identify all actual positives. A recall of 29% means that the model correctly identifies 29% of all actual renters. This is relatively low, indicating that the model misses out on a large number of potential renters (71% are missed). This could mean the model is too conservative in predicting who will rent.

- **Model F1 Score (0.44 or 44%)**: The F1 score is the harmonic mean of precision and recall, providing a single measure to balance both the precision and the recall. An F1 score of 44% indicates a moderate balance between precision and recall, which in this case leans more towards high precision and low recall. This lower score suggests that the model, while accurate in its positive predictions, is not capturing a significant portion of the actual positive cases.

### Insights from the Metrics

- **High Precision, Low Recall**: The combination of high precision and low recall suggests that the model is very cautious about predicting that a lead will rent. It prefers to miss potential renters rather than falsely predict that someone will rent. This can be useful in scenarios where false positives (predicting a rental when it won't happen) have higher consequences than false negatives (missing out on potential renters).

- **Possible Class Imbalance**: This pattern of results might also indicate a class imbalance in the dataset, where there are fewer actual renters than non-renters. Such an imbalance could make the model biased towards predicting the majority class (non-renters in this case).

- **Potential for Model Improvement**: There is a need to improve the model’s recall without substantially sacrificing precision. Techniques such as adjusting the decision threshold, using cost-sensitive learning, or rebalancing the dataset might help increase recall.

- **Practical Application Considerations**: Depending on the business context, you may need to adjust the model to either minimize false positives or to capture more true positives. For instance, if missing out on a potential renter has a higher business cost than targeting non-renters, you might opt to improve recall even if it slightly lowers precision.

Overall, these metrics should guide further refinement and validation of the model, ensuring it aligns with business objectives and operational strategies.

### Conclusion
The logistic regression analysis highlights that the number of virtual tours is a key predictor of conversion, emphasizing the importance of interactive and engaging online experiences for potential renters. However, other factors like the number of website visits and session lengths do not show significant effects, indicating that quality interactions might outweigh sheer quantity in driving conversions. This insight can inform targeted strategies to optimize marketing efforts, focusing on quality engagement over simply driving higher traffic.
