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
x = f(z; \theta), ~~ z \sim p(z; \psi)
$$

where $f(\cdot; \theta)$ is a mapping function from $z$ to $x$ parametrized by $\theta$ and p(z; \psi) is the base (sometimes also refered as prior) distribution parametrized by $\psi$.
The defining propertires of a normalizing flow are:
 - $f(\cdot; \theta)$ must be invertible;
 - $f(\cdot; \theta)$ and $f^{-1}(\cdot: \theta)$ must be differentiable.

Under such constraint the density of $x$ is well-defined thanks to the change-of-variable theorem [[2]](#ref:change-of-variable):

$$
\begin{align*}
    \int p(x) \partial x &= \int p(z) \partial z = 1 \\
    \implies p(x) & = p(z) \cdot |\frac{\partial z}{\partial x}| \\
    & = p\big(f^{-1}(x; \theta)\big) \cdot |\frac{\partial f^{-1}(x; \theta)}{\partial x}| \\
    % & = p\big(f^{-1}(x; \theta)\big) \cdot |f^{-1'}(x; \theta)|.
\end{align*}
$$

By definition $\partial x$ represents the width of an infinitesimally small rectangle with heigh $p(x)$; thus $\frac{\partial f^{-1}(x; \theta)}{\partial x}$ is the ratio between the area of the rectangles defined in two different cordination system one in $x$ and one in $z$.
For example, Fig. [1]($fig:change-of-variable) shows how the affine transformation $f^{-1}(x) = (5 \cdot x) - 2$ map the Normal distribution $p(x; \mu=0, \sigma=1)$ into another Gaussian distribution $p(z; \mu=100, \sigma=5)$.
However, since $\frac{\partial z}{\partial x} = 5$ the area $\partial x$ get streched by a factor of 5 when transformed into the variable $z$. Thus, $p(z)$ needs to be reduced by a factor of 5 in order to be a valid probability density function $(\int p(z) \partial z = 1)$:

$$
p(z) = \frac{p(x)}{\frac{\partial f^{-1}(x)}{\partial x}} = \frac{p(x)}{f^{-1'}(x)}.
$$


<div style="text-align:center;" id="fig:change-of-variable">
    <figure>
        <img src="{{site.baseurl}}/assets/img/norm_flow/change-of-variable.png" style="max-width: 98%">
        <figcaption style="font-size:small;">
            Figure 1: Example of change-of-variable. The random variable $x$ is converted in to another random variable $z$ by means of the affine function $f^{-1}(x) = 5x - 2$; in other words $z=5x-2$.
            To be a valid density function $p(z)$ needs to satisfy the property $\int p(z) \partial z = 1$ however as the transformation $f^{-1'}(x)$ streches the space by a factor of 5, we need to dereduce the density by the same amount.
        </figcaption>
    </figure>
</div>

<!-- add multidimensional chenge of variable and jacobians/determinant -->
# Credits

The content of this post is based on the lectures and code of [Pieter Abbeel](https://sites.google.com/view/berkeley-cs294-158-sp20/home), [Justin Solomon](https://groups.csail.mit.edu/gdpgroup/6838_spring_2021.html) and [Karpathy's](https://github.com/karpathy/pytorch-normalizing-flows) tutorial.
Moreover, I want to credit [Lil'Long](https://lilianweng.github.io/posts/2018-10-13-flow-models/) and [Eric Jang](https://blog.evjang.com/2018/01/nf1.html) for their amazing tutorials.

# Refences

<ol>
    <li id="ref:normalization-flow-review"> Papamakarios, G., Nalisnick, E., Rezende, D. J., Mohamed, S., & Lakshminarayanan, B. (2019). Normalizing Flows for Probabilistic Modeling and Inference. <a href="http://arxiv.org/abs/1912.02762">arxiv.org/abs/1912.02762</a></li>
    <li id="ref:change-of-variable"> Weisstein, Eric W. "Change of Variables Theorem." From MathWorld--A Wolfram Web Resource. <a href="https://mathworld.wolfram.com/ChangeofVariablesTheorem.html">mathworld.wolfram.com/ChangeofVariablesTheorem.html</a> </li>
    
</ol>