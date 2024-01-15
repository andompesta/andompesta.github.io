---
layout: post
mathjax: true
title:  "Traditional Machine Learning"
author: "Sandro Cavallari"
tag: "Interviews"
---



# Linear Regression

Linear regression is one of the most used and simple ML model.
It expresses a linear relationship between a dependendt variable (y-axe) and one or multiple indipendent variables (x-axe).
THe basic formilation of a linear regression can be described as follows:

$$
Y = \epsilon + w_1 X_1 + ... + w_n X_n
$$

where:
- $Y$ is the dependent variable, also known as the response or target or outcome variable.
- $X_i$ are indipendent variabels, also known as predictors or features.
- $w_i$ are the coefficients of the independent variables. These coefficients represent the change in the dependent variable for a one-unit change in the respective independent variable, assuming all other variables are held constant.
- $\epsilon$ is the bias term, which account for the viariability in $Y$ that can not be explained by linear model.

The learning process adjust the model's coefficients in such a way that we minimise the sum of the squared residuals:

$$
\mathcal{L}(Y, \hat{Y}) = (Y - \hat{Y})^2
$$

where $\mathbf{Y}$ is the predicted value and $\hat{\mathbf{Y}}$ is the actual value for a given training example.
Due to this loss formulation linear models are also known as least square estimators.
The training process adjust the model's parameters $\theta$ so to minimise the loss:

$$
\frac{\partial \mathcal{L}(Y, \hat{Y}; \theta)}{\partial \theta}.
$$


Linear regression has several key assumptions:
1. Linearity between the indipendent and dependent variables.
2. Independence across predictors. This is important for stability and interpretability reasons. Correlated variables (colinearity) can cause the model to give big changes in the outcome-variable for small changes in the predictors variables, braking the assumtion of a liear relationship. Moreover, a model with indipendent variables is easier to interprate as each variable has unique contribution to the prediction.
3. Homoscedasticity of residuals. If residuals are not normally distributed with constant variance it is difficult to ensure that the model is not biased and it is difficult to do error-analysis.

Ensuring these assumptions are met is crucial for the reliability and validity of the regression analysis.

## Robust Linear Model


Most of the time real-world dataset are sunbject to multi-colinearity among indipendent variables.
A good approach to alleviate the multi-colinearity problems is to use the [**Ridge Regression**](https://www.youtube.com/watch?v=Q81RR3yKn30&ab_channel=StatQuestwithJoshStarmer):

$$
Y = \epsilon + w_1 X_1 + ... + w_n X_n + \lambda \sum_{1 \leq i \leq n} w_i^2.
$$

Note that Ridge Regression is equal to the linear regression plus a discount factor.
The discount factor is composed by the square of the model's weights; thus we are minimising the square of the model weight.
This penalty factor inrease the model bias, but reduces the prediction variance by imposing a normal prior to the model parameters.


----

[**Lasso Regression**](https://www.youtube.com/watch?v=NGf0voTMlcs&ab_channel=StatQuestwithJoshStarmer) is another popular variant of the linear model.
Like Ridge regression, Lasso regression add a penalty term to the linear model loss function; however, lasso regression minimise the absolute value of the parameters rather than square value:

$$
Y = \epsilon + w_1 X_1 + ... + w_n X_n + \lambda \sum_{1 \leq i \leq n} |w_i|.
$$

The difference is that lasso regression generate a sparse model as can push model's parameters all the way to 0, while ridge regression only makes them really small.

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