---
layout: post
mathjax: true
title:  "Normalization Flow"
author: "Sandro Cavallari"
tag: "Deep Learning"
comments_id: 9
---

Normalizing Flows (NF)[[1]](#ref:normalization-flow-review) are a powerful thecnique that allows to learn and sample from complex probability distributions.
They are known to be generative models that allow for exact likelihood estimation of real input data $p(x)$.
Instead of relaing on approximation like in variational inference, normalizing flow operate by feed a simple distribution from which it is possible sample from $p(z)$ trought a series of maps to produce a richer distribution:

$$
x = f_{\theta}(z), ~~ z \sim p(z; \psi)
$$

where $f_{\theta}(\cdot)$ is a mapping function from $z$ to $x$ parametrized by $\theta$ and $p(z; \psi)$ is the base (sometimes also refered as prior) distribution parametrized by $\psi$.
The defining propertires of a normalizing flow are:
 - $f_{\theta}(\cdot)$ must be invertible;
 - $f_{\theta}(\cdot)$ and $f_{\theta}^{-1}(\cdot)$ must be differentiable.

Under such constraint the density of $x$ is well-defined thanks to the change-of-variable theorem [[2]](#ref:change-of-variable):

$$
\begin{align*}
    \int p(x) \partial x &= \int p(z) \partial z = 1 \\
    \implies p(x) & = p(z) \cdot |\frac{\partial z}{\partial x}| \\
    & = p\big(f_{\theta}^{-1}(x)\big) \cdot |\frac{\partial f_{\theta}^{-1}(x)}{\partial x}| \\
    % & = p\big(f^{-1}(x; \theta)\big) \cdot |f^{-1'}(x; \theta)|.
\end{align*}
$$

By definition $\partial x$ represents the width of an infinitesimally small rectangle with heigh $p(x)$; thus $\frac{\partial f_{\theta}^{-1}(x)}{\partial x}$ is the ratio between the area of the rectangles defined in two different cordination system one in $x$ and one in $z$.
For example, Fig. [1]($fig:change-of-variable) shows how the affine transformation $f_{\theta}^{-1}(x) = (5 \cdot x) - 2$ maps the Normal distribution $p(x; \mu=0, \sigma=1)$ into another Gaussian distribution $p(z; \mu=100, \sigma=5)$.
As $\frac{\partial z}{\partial x} = 5$ the area $\partial x$ get streched by a factor of 5 when transformed into the variable $z$. Thus, $p(z)$ needs to be reduced by a factor of 5 in order to be a valid probability density function (every probability density needs to satisfy $\int p(z) \partial z = 1$):

$$
p(z) = \frac{p(x)}{\frac{\partial f_{\theta}^{-1}(x)}{\partial x}} = \frac{p(x)}{f_{\theta}^{-1'}(x)}.
$$

<div style="text-align:center;" id="fig:change-of-variable">
    <figure>
        <img src="{{site.baseurl}}/assets/img/norm_flow/change-of-variable.png" style="max-width: 98%">
        <figcaption style="font-size:small;">
            Figure 1: Example of change-of-variable. The random variable $x$ is converted in to another random variable $z$ by means of the affine function $f_{\theta}^{-1}(x) = 5x - 2$; in other words $z=5x-2$.
            To be a valid density function $p(z)$ needs to satisfy the property $\int p(z) \partial z = 1$ however as the transformation $f_{\theta}^{-1'}(x)$ streches the space by a factor of 5, we need to reduce the density by the same amount.
            To this end note the difference between the max value of $p(z)$ and $p(x)$.
            The picture in the bottom left gives a visual representation of how $\partial x$ get strached by the transformation $f_{\theta}^{-1}(\cdot)$.
        </figcaption>
    </figure>
</div>


In the previous paragraph we introduced the concept of volume-preserving transformations; following the same reasoning, it is possible to extend the same concept to the multidimentional space by considering $\frac{\partial z}{\partial x}$ not as a simple derivative, but rather as the **Jacobian** matrix:

$$
J_{z}(x) = \begin{bmatrix} 
    \frac{\partial z_1}{\partial x_1} & \dots & \frac{\partial z_1}{\partial x_D}\\
    \vdots & \ddots &  \vdots \\
    \frac{\partial z_D}{\partial x_1} & \dots & \frac{\partial z_D}{\partial x_D}
\end{bmatrix},
$$

and the difference in areas as diffence in volumes quantified by the **determinant** of the Jacobian matrix
$det(J_{z}(x)) \approx \frac{Vol(z)}{Vol(x)}$.
Putting everithing togheter we can formalize the normalization flow:

$$
\begin{align*}
p(x) & = p(z) \cdot |det(\frac{\partial z}{\partial x})| \\
& = p(f_{\theta}^{-1}(x)) \cdot |det(\frac{\partial f_{\theta}^{-1}(x)}{\partial x})| \\
& = p(f_{\theta}^{-1}(x)) \cdot |det(J_{f_{\theta}^{-1}}(x))|.
\end{align*}
$$

### Flow as Finate Composition of Transformations

The term flows derives from the fact that in the general case the transformations $f_{\theta}(\cdot)$ and $f_{\theta}^{-1}(\cdot)$ are defined as a finite compositions of simpler transformations $f_{\theta_i}$:

$$
\begin{align*}
x & = z_{K} = f_{\theta}(z_0) = f_{\theta_K} \dots f_{\theta_2} \circ f_{\theta_1}(z_0) & \\
p(x) & = p_K(z_{k}) = p_{K-1}(f_{\theta_K}^{-1}(z_{k})) \cdot \Big| det\Big(J_{f_{\theta_K}^{-1}}(z_{k})\Big)\Big| & \\
& = p_{K-1}(z_{K-1}) \cdot \Big| det\Big(J_{f_{\theta_K}^{-1}}(z_k)\Big)\Big| & \text{Due to the definition of } f_{\theta_K}(z_{K-1}) = z_K\\
& = p_{K-1}(z_{K-1}) \cdot \Big| det\Big( J_{f_{\theta_K}(z_{K-1})} \Big)^{-1}\Big| & \text{As: } J_{f_{\theta_K}^{-1}}(x) = \frac{f_{\theta_K}^{-1}(z_k)}{\partial z_K} \\
& & = \Big(\frac{\partial z_K}{\partial f_{\theta_k}^{-1}(z_k)} \Big)^{-1} \\
& & = \Big(\frac{\partial f_{\theta_{k}}(z_{K-1})}{\partial z_{K-1}}\Big)^{-1} \\
& & = \Big( J_{f_{\theta_K}(z_{K-1})} \Big)^{-1} \\
& = p_{K-1}(z_{K-1}) \cdot \Big| det\Big( J_{f_{\theta_K}}(z_{K-1}) \Big)\Big|^{-1}. \\
\end{align*}
$$

Eventually, the very same process can then be extended to all step $i$ and obtain the final definition:

$$
p(x) = p(z_0) \cdot \prod_{i=1}^k \Big| J_{f_{\theta_i}}(z_{i-1}) \Big|^{-1}.
$$
<!-- complexity of daterminat computation -->

### Training Procedures and Inference

<!-- training process with max-likelihood -->
<!-- KL divergence formulation -->
<!-- comparison of inference with other models -->
<!-- tractable inference and sampling -->
# Credits

The content of this post is based on the lectures and code of [Pieter Abbeel](https://sites.google.com/view/berkeley-cs294-158-sp20/home), [Justin Solomon](https://groups.csail.mit.edu/gdpgroup/6838_spring_2021.html) and [Karpathy's](https://github.com/karpathy/pytorch-normalizing-flows) tutorial.
Moreover, I want to credit [Lil'Long](https://lilianweng.github.io/posts/2018-10-13-flow-models/) and [Eric Jang](https://blog.evjang.com/2018/01/nf1.html) for their amazing tutorials.

# Refences

<ol>
    <li id="ref:normalization-flow-review"> Papamakarios, G., Nalisnick, E., Rezende, D. J., Mohamed, S., & Lakshminarayanan, B. (2019). Normalizing Flows for Probabilistic Modeling and Inference. <a href="http://arxiv.org/abs/1912.02762">arxiv.org/abs/1912.02762</a></li>
    <li id="ref:change-of-variable"> Weisstein, Eric W. "Change of Variables Theorem." From MathWorld--A Wolfram Web Resource. <a href="https://mathworld.wolfram.com/ChangeofVariablesTheorem.html">mathworld.wolfram.com/ChangeofVariablesTheorem.html</a> </li>
    
</ol>