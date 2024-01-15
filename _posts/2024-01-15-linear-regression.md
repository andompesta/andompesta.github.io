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

It might be the case that we are interested in understanding the relationship between the dependent variable and one of our predictors.
Precisly we are interested in understadining if a parameter $w_i$ has a significant relationship with the response variable $Y$, in other words we are interessted in understanding if there is a singificant reduction in the loss if we add the predictor $X_i$.

Formally we are interested in test the following hypothesis:

$$
H_0 : w_i = 0 \\
H_1 : w_i \neq 0.
$$

With an abuse in notation let us define $L(Y, \hat{Y}: \theta_{\not i})$ and $L(Y, \hat{Y}: \theta_{i})$ as the sum of squared residuals of a model respectively without and with the i-predictor.
Under the assumption of indipendence and homoscedasticity of the model's parameters, it is possible to compute estimate the significance of the i-predictor by performing the t-test:

$$
F = \frac{ \frac{L(Y, \hat{Y}: \theta_{\not i}) - L(Y, \hat{Y}: \theta_{i})}{p_2} }{ \frac{L(Y, \hat{Y}: \theta_{i})}{n - p} }
$$

where $p$ and $p_2$ are the degree of freedom of the overal model and the model with only $i$-th predictor; while $n$ is the number of training examples.


The numerator of $F$ is the reduction in the residual sum of squares per degree of freedom spent.
The denominator is the average residual sum of squares, a measure of noise in the model.
Thus, an $F$-ratio of one would indicate that the variables in are just adding noise.
A ratio in excess of one would be indicative of signal.
We usually reject $H_0$, and conclude that the i-th variable have an effect on the response if the $F$-criterion exceeds the 95-th percentage point of the $F$-distribution with $p_2$ and $(n-p)$ degrees of freedom.


​
 


## Residual Plot


How to minimise the residuals in a regression task?

    1) Feature Selection: Choose relevant and significant variables to include in the model.
    2)Transformation of Variables: Apply transformations (like log, square root, or inverse) to make the relationship more linear.
    3)Polynomial Regression: Use higher-order terms (quadratic, cubic, etc.) if the relationship is not purely linear.
    2) Interaction Terms: Include interaction terms if the effect of one variable depends on another.
    3) Regularization Techniques: Methods like Ridge, Lasso, or Elastic Net can help in reducing overfitting and improving prediction.
    6)Residual Plots: Use residual plots to check for non-linearity, unequal error variances, and outliers.
    7)Influence Measures: Identify and investigate influential observations that might disproportionately affect the model's performance.
    8)Homoscedasticity Testing: Ensure that residuals have constant variance across different levels of predictors.