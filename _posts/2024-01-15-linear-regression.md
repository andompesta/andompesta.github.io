---
layout: post
mathjax: true
title:  "Traditional Machine Learning"
author: "Sandro Cavallari"
tag: "Interviews"
---



# Linear Regression

Linear regression is a widely utilized and straightforward machine learning model. 
It establishes a linear relationship between a dependent variable (represented on the y-axis) and one or more independent variables (represented on the x-axis). 
The fundamental formulation of linear regression is as follows:

$$
Y = \epsilon + w_1 X_1 + ... + w_n X_n
$$

where:
- $Y$ is the dependent variable, also referred to as the response, target, or outcome variable.
- $X_i$ represents the independent variables, also known as predictors or features.
- $w_i$ are the coefficients corresponding to the independent variables. These coefficients indicate the expected change in the dependent variable for a one-unit change in the respective independent variable, with all other variables held constant.
- $\epsilon$ is the bias term, which account for the viariability in $Y$ that can not be explained by linear model.

The learning process involves adjusting the model's coefficients to minimize the sum of the squared differences between the observed and predicted values:

$$
\mathcal{L}(Y, \hat{Y}) = (Y - \hat{Y})^2
$$

Here, $\mathbf{Y}$ is the predicted value and $\hat{\mathbf{Y}}$ is the actual value for a given training example. This loss formulation is why linear models are often referred to as least squares estimators. The training process adjusts the model's parameters $\theta$ to minimize this loss:

$$
\frac{\partial \mathcal{L}(Y, \hat{Y}; \theta)}{\partial \theta}.
$$

Linear regression relies on several key assumptions:

1. Linearity: There must be a linear relationship between the independent and dependent variables.
2. Independence: The predictors should be independent of each other. This is vital for the model's stability and interpretability. Collinearity, or correlation between variables, can lead to significant changes in the outcome variable for minor alterations in the predictor variables, contradicting the assumption of a linear relationship. Additionally, a model with independent variables is easier to interpret as each variable contributes uniquely to the prediction.
3. Homoscedasticity: The residuals should be normally distributed with constant variance. Without this, it becomes challenging to ensure that the model is unbiased and to conduct accurate error analysis.

Verifying these assumptions is crucial for ensuring the reliability and validity of the regression analysis.

## Robust Linear Model

In the real world, datasets often suffer from multicollinearity among independent variables. One effective method to mitigate this issue is through [**Ridge Regression**](https://www.youtube.com/watch?v=Q81RR3yKn30&ab_channel=StatQuestwithJoshStarmer), described by the formula:

$$
Y = \epsilon + w_1 X_1 + ... + w_n X_n + \lambda \sum_{1 \leq i \leq n} w_i^2.
$$

Ridge Regression is essentially linear regression augmented with a penalty term. This term comprises the squared coefficients of the model, effectively minimizing their magnitude. This added penalty increases the model's bias but decreases prediction variance by imposing a normal distribution prior on the model parameters.


----

On the other hand, [**Lasso Regression**](https://www.youtube.com/watch?v=NGf0voTMlcs&ab_channel=StatQuestwithJoshStarmer) is another widely-used variation of linear regression. Similar to Ridge, Lasso adds a penalty to the loss function of the linear model. However, Lasso minimizes the absolute value of the coefficients rather than their square:

$$
Y = \epsilon + w_1 X_1 + ... + w_n X_n + \lambda \sum_{1 \leq i \leq n} |w_i|.
$$

The key distinction is that Lasso regression can reduce some coefficients to zero, producing a sparse model, whereas Ridge regression only reduces them to near zero.

In conclusion, both these regularization approaches generate a model that has a bigger bias, but better generalization capability.

## Parameter's Analysis

There are instances when we seek to comprehend how one of our predictors influences the dependent variable. Specifically, our interest lies in determining whether the parameter $w_i$ significantly affects the response variable $Y$ - that is, whether including the predictor $X_i$ leads to a notable reduction in the model's loss.

Formally, this involves testing the following hypotheses:

$$
H_0 : w_i = 0 \\
H_1 : w_i \neq 0.
$$

For clarity, let's denote $L(Y, \hat{Y}: \theta_{\not i})$ and $L(Y, \hat{Y}: \theta_{i})$ as the sum of squared residuals for models excluding and including the $i$-th predictor, respectively. Assuming independence and homoscedasticity of the model's parameters, the significance of the $i$-th predictor can be assessed using the F-test:

$$
F = \frac{ \frac{L(Y, \hat{Y}: \theta_{\not i}) - L(Y, \hat{Y}: \theta_{i})}{p_2} }{ \frac{L(Y, \hat{Y}: \theta_{i})}{n - p} } .
$$

Here, represent the degrees of freedom for the overall model and the model containing only the $i$-th predictor; while $n$ is the number of training examples.


The numerator of $F$-test represents the reduction in the residual sum of squares per additional degree of freedom utilized. The denominator is an estimate of the residual variance, serving as a measure of the model's inherent noise. An $F$-ratio of one suggests that the predictors merely contribute noise. A ratio greater than one implies meaningful contribution, or signal, from the predictors. Typically, we reject $H_0$ and conclude that the $i$-th variable significantly impacts the response if the $F$-statistic exceeds the 95th percentile of the $F$-distribution with $p_2$ and $(n-p)$ degrees of freedom. A full derivation of this result is available [here](https://grodri.github.io/glms/notes/c2s3).
 
## R-squared
The $R^2$ metric, also known as the coefficient of determination, is a statistical measure used to assess the goodness of fit of a regression model. 
It represents the proportion of the variance in the dependent variable that is predictable from the independent variables. 
In simpler terms, $R^2$ indicates how well the data fit the regression model (the closer the value of $R^2$ is to 1, the better the fit) and can be computed as:

$$
R^2 = \frac{\mathcal(Y, \hat{Y}; \varnothing) - \mathcal(Y, \hat{Y}; \theta)}{\mathcal(Y, \hat{Y}; \varnothing)}
$$

here $\mathcal(Y, \hat{Y}; \varnothing)$ represent the sum of squared residuals of a model wihtout parameters, a.k.a a model that alwasy predict the mean of the response variable.
Similarly, $\mathcal(Y, \hat{Y}; \theta)$ is the linear regression developed.

The $R^2$ value ranges from 0 to 1. A vale of 0 means that the model does not explain any of the variability of the response data around its mean.
On the other hand, an $R^2$  of 1 indicates that the model explains all the variability of the response data around its mean.
In conclusion $R^2$ is a simple to compute, yet informative metric to determin how much our variance our model is able to predict correctly.

## Improvements

In general it is possible to rank the model performance in terms of $\mathcal{L}$.
Thus, here are useful methods to reduce the sum of squared residuals:

1) Feature Selection: Choose relevant and significant variables to include in the model.

2) Transformation of Variables: Apply transformations (like log, square root, or inverse) to make the relationship more linear.

3) Polynomial Regression: Use higher-order terms (quadratic, cubic, etc.) if the relationship is not purely linear.
   
4) Interaction Terms: Include interaction terms if the effect of one variable depends on another. An interaction variable in a linear model represents the combined effect of two or more variables on the dependent variable, where the effect of one variable depends on the level of another variable. In other words, it's used to capture situations where the relationship between a predictor and the outcome changes based on the value of another predictor. Formally, an interaction variable in alinear model is defined as:

$$
Y = \epsilon + w_1 X_1 + ... + w_n X_n + w_{n+1} (X_i \cdot X_j)
$$

5) Regularization Techniques: Methods like Ridge, Lasso, or Elastic Net can help in reducing overfitting and improving prediction.
6) Residual Plots: Use residual plots to check for non-linearity, unequal error variances, and outliers.
7) Influence Measures: Identify and investigate influential observations that might disproportionately affect the model's performance.
8) Homoscedasticity Testing: Ensure that residuals have constant variance across different levels of predictors.


# Residual Plots

Residuals plots are a type of diagrams that has the residuals on the y-axes and the predicted value on x-axes. The residuals should have a uniform distribution.
These types of plots are usefull to assess the performance of the model as well as assess how clean the dataset is. Namelly residuals plot can be used for:

1. Checking Homoscedasticity: A key assumption in linear regression is that the residuals have constant variance at every level of the independent variable(s) (homoscedasticity). If the residuals fan out or form a pattern as the predicted values increase, this indicates heteroscedasticity, which can lead to inefficient estimates and affect hypothesis testing.

2. Identifying Non-Linearity: If the relationship between the variables is not linear, the residuals will often display a systematic pattern. A residuals plot can help identify such non-linear relationships, suggesting that a different model or a transformation of variables might be more appropriate.

3. Detecting Outliers: Residuals plots can reveal outliers – points that have a much larger error than the other observations. These outliers can disproportionately influence the model fit and can be indicative of data entry errors, or that the outlier is from a different population.

4. Evaluating Model Fit: If a model is a good fit for the data, the residuals should be randomly scattered around zero. Patterns or trends in the residuals suggest that the model is not capturing some aspect of the data, which could lead to biased or inaccurate predictions.

5. Checking Independence of Residuals: For a good model, the residuals should be independent of each other. If there's a pattern over time (in time series data) or a pattern associated with another variable, this is a sign that the model is missing a key piece of information.

6. Verifying Normal Distribution of Residuals: In many forms of regression analysis (like linear regression), it's assumed that residuals are normally distributed. By plotting the residuals and visually checking their distribution (or using statistical tests), we can validate this assumption. A common method is to use a Q-Q plot (quantile-quantile plot) to compare the distribution of the residuals to a normal distribution.