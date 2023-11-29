---
layout: post
mathjax: true
title:  "Normalizing Flows"
author: "Sandro Cavallari"
tag: "Deep Learning"
comments_id: 9
---

Normalizing Flows (NF) represent a potent technique that facilitates the learning and sampling from intricate probability distributions [[1]](#ref:normalization-flow-review) [[8]](#ref:nice). These models, categorized as generative models, enable the precise estimation of likelihood for continuous input data, denoted as $p(x)$. In contrast to methods such as variational inference that rely on approximations, normalizing flows function by transforming samples from a simple distribution, denoted as $z \sim p(z)$, into samples from a more complex distribution using the following transformation:

$$
x = f_{\theta}(z), ~~ z \sim p(z; \psi).
$$

Here, $f_{\theta}(\cdot)$ is a mapping function from $z$ to $x$, parametrized by $\theta$, and $p(z; \psi)$ is the base distribution (sometimes referred to as the prior distribution), parametrized by $\psi$, from which samples can be drawn. The essential properties defining a normalizing flow include:

The defining propertires of a normalizing flow are:
 - $f_{\theta}(\cdot)$ must be invertible.
 - Both $f_{\theta}(\cdot)$ and $f_{\theta}^{-1}(\cdot)$ must be differentiable.

Adhering to these constraints ensures the well-defined density of $x$, as established by the change-of-variable theorem [[2]](#ref:change-of-variable):

$$
\begin{align*}
    \int p(x) \partial x &= \int p(z) \partial z = 1 \\
    \implies p(x) & = p(z) \cdot |\frac{\partial z}{\partial x}| \\
    & = p\big(f_{\theta}^{-1}(x)\big) \cdot |\frac{\partial f_{\theta}^{-1}(x)}{\partial x}| \\
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


In the previous paragraph we introduced the concept of area-preserving transformations; following the same reasoning, it is possible to extend this concept to the multidimentional space by considering $\frac{\partial z}{\partial x}$ not as a simple derivative, but rather as the **Jacobian** matrix:

$$
J_{z}(x) = \begin{bmatrix} 
    \frac{\partial z_1}{\partial x_1} & \dots & \frac{\partial z_1}{\partial x_D}\\
    \vdots & \ddots &  \vdots \\
    \frac{\partial z_D}{\partial x_1} & \dots & \frac{\partial z_D}{\partial x_D}
\end{bmatrix}.
$$

In the multidimentional setting the difference in areas became diffence in volumes quantified by the **determinant** of the Jacobian matrix
$det(J_{z}(x)) \approx \frac{Vol(z)}{Vol(x)}$.
Putting everithing togheter we can formalize a miltidimentional normalization flow as:

$$
\begin{align*}
p(x) & = p(z) \cdot |det(\frac{\partial z}{\partial x})| \\
& = p(f_{\theta}^{-1}(x)) \cdot |det(\frac{\partial f_{\theta}^{-1}(x)}{\partial x})| \\
& = p(f_{\theta}^{-1}(x)) \cdot |det(J_{f_{\theta}^{-1}}(x))|.
\end{align*}
$$

## Generative Process as Finate Composition of Transformations

In the general case the transformations $f_{\theta}(\cdot)$ and $f_{\theta}^{-1}(\cdot)$ are defined as finite compositions of simpler transformations $f_{\theta_i}$:

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

In so doing, $p(z_i)$ is fully described by $z_{i-1}$ and $f_{\theta_i}$, thus it is possible to extended the previous reasoning to all i-steps of the overall generative process:

$$
\begin{equation}
p(x) = p(z_0) \cdot \prod_{i=1}^k \Big| det \big( J_{f_{\theta_i}}(z_{i-1}) \big) \Big|^{-1}.
\label{eq:flow_generator}
\end{equation}
$$

Note that $f_{\theta}$, in the contex of generative models, is also referd as a pushforwartd mapping from a simple density $p(z)$ to a more complex $p(x)$.
The inverse transfomration $f_{\theta}^{-1}$ is instead called the normalization function as it "normalizes"
 a complex distribution into a simpler one, one step at a time.  


## Training Procedures

As overmentioned NF are efficent models that allow sampling and learning complex distributions.
Thus, the most common application for NF are density estimation and data generation.  
On the one hand, density estimation is an handy task when someone is intersted in computing statistical quantities over unseen data. For example, [[3]](#ref:density-estimation) and [[4]](#ref:ffjord) demonstrate that NF models are able to estimate densities over tabular and image datasets. 
Moreover, density estimation is the base capabilities that allows NF to be adopted for anomaly detection [[5]](#ref:nf-anomaly-detection) while it requires carefuly tuning for out-of-distribution detection [[6]](#ref:nf-for-odd).  
On the other hand, the main application for NF is related to data generation. As abote mentioned, under some mild assumtions,  NFs are capable of sampling new datapoints from a complex distribution $p(x)$. [[7]](#ref:glow) is a primal example of NF applied to image generation, while [[9]](#ref:wave-net) and [[10]](#ref:flow-wave-net) demonstrate that NF can sussesfully learn audio signals.


One of the main advantages of NFs over other probabilistic generative model is that they can be easily trained by minimasing some divergence metric between $p(x; \theta)$ and the target distribution $p(x)$.
In most of the cases NF are trained by minimasing the KL-diverngence between the two distributions:

$$
\begin{align*}
    \mathcal{L}(\theta) & = D_{KL}[p(x) || p(x; \theta)] \\
        & = - \sum_{x \sim p(x)} p(x) \cdot \log \frac{p(x; \theta)}{p(x)} \\
        & = - \sum_{x \sim p(x)} p(x) \cdot \log p(x; \theta) + \sum_{x \sim p(x)} p(x) \cdot \log p(x) \\
        & = - \mathbb{E}_{x \sim p(x)} \Big[ \log p(x; \theta) \Big] + \mathbb{E}_{x \sim p(x)} \Big[ \log p(x) \Big] \\
        & = - \mathbb{E}_{x \sim p(x)}\Big[\log p(x; \theta)\Big] + const. ~~~ \text{As it does not depend on $\theta$} \\
        & = - \mathbb{E}_{x \sim p(x)}\Big[\log p\big(f_{\theta}^{-1}(x)\big) + \sum_{i=1}^{K} \log \Big| det\big( J_{f_{\theta_i}^{-1}}(z_{i}) \big)\Big| \Big] + const.
\end{align*}
$$

where $p(f_{\theta}^{-1}(x)) = p(z_0)$ and $z_K$ is equal to $x$. Given a fixed training set $X_N = \\{ x_n \\}_{n=1}^N$ the above loss reduces to the negative log-likelihood usually optimized by stocastic gradient descent:

$$
\begin{equation}
\mathcal{L}(\theta) = - \frac{1}{N} \sum_{n=1}^N \log p\big(f_{\theta}^{-1}(x)\big) + \sum_{i=1}^{K} \log \Big| det\big( J_{f_{\theta_i}^{-1}}(z_{i}) \big)\Big|.
\label{eq:flow_loss}
\end{equation}
$$

Note that the loss function (Eq. $\eqref{eq:flow_loss}$) is computed starting from a datapoint $x$ and revert it to a plausable latent variable $z_0$. In so doing, the structural formulation of $p(z_0)$ is the major factor that define the training signals: if $p(z_0)$ is too loose, the training process does not have much to learn; if it is too strich, the training process might be too difficlt to learn.  
Moreover, the training process is exactly the inverse of the generative process defined in Eq. $\eqref{eq:flow_generator}$; thus the sum of determinants.  
Finally, to achieve a computationally efficent training there is the need to efficently compute the determinants of $J_{f_{\theta_i}^{-1}}$.
While it is possible to leverage auto-diff libraries to compute the gradiens with respect to $\theta_i$ of the Jacobians matrix and its determinant such computation is expencive ($O(n)^3$); thus a large amount of research when into designing transformations that have efficent Jacobian determinant formulations.

### Training Example

As overmentioned the training process of a NF is based on the mapping between a given input data $x$ to a particular base distribution $p(z_0)$. Usually, the base distribution is a well-known distribution such as multivariate gaussian, uniform or any other exponential destirbution. Similarly the mapping function it is usually implemented as a neural network.
Starting from the first-principle: we can specify any NF model as composed of a base distribution and a series of flow that map $x$ to $z_0$:

```python
class NormalizingFlow(nn.Module):
    def __init__(
        self,
        prior: Distribution,
        flows: nn.ModuleList,
    ):
        super().__init__()
        self.prior = prior
        self.flows = flows

    def forward(self, x: Tensor):
        bs = x.size(0)
        zs = [x]
        sum_log_det = torch.zeros(bs).to(x.device)
        for flow in self.flows:
            z = flow(x)
            log_det = flow.log_abs_det_jacobian(x, z)
            zs.append(z)
            sum_log_det += log_det
            x = z

        prior_logprob = self.prior.log_prob(z).view(bs, -1).sum(-1)
        log_prob = prior_logprob + sum_log_det

        intermediat_results = dict(
            prior_logprob=prior_logprob,
            sum_log_det=sum_log_det,
            zs=zs,
        )
        return log_prob, intermediat_results
```
where the prior can be any base distribution inplemented in [torch.distributions](https://pytorch.org/docs/stable/distributions.html) and flows can be any module that statisfy the NF properties.

Supposed we are given a 1D dataset as shown in Fig. [[2.a]](#fig:1d_dataset), we can fit a NF the underling probability distribution $p(x)$ of the given dataset.
To successfully learn the density of the dataset, we need a based distribution, let say a Beta distribution parametrized by $\alpha = 2$ and $\beta = 5$, and a functional definition for our flow, in this case a Gaussian Mixture Model with 4 different component.

```python
model = NormalizingFlow(
    prior=Beta(2.0, 5.0),
    flows=nn.ModuleList([
        GMMFlow(n_components=4, dim=1)
    ]),
)
optimizer = optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-5)
```

Given these ingridients we can train the model by minimizing the negative log likelihood by SGD:
```python
for epoch in range(epochs):
    model.train()

    for idx, (x, y) in enumerate(dataloader):
        optimizer.zero_grad()
        log_prob, _ = model(x)
        loss = -log_prob.mean()  # nll
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
```
Note that, no labels is used for training, as the objective is to directly maximise the predicted density of the dataset.
Fig. [[2.b]](#fig:1d_dataset) shows the learned density and how the 4 different components are used to correctly model $p(x)$.
While the same results might be achieved by using only 2 components, in the general case the minimum number of needed components is not needed apriori; thus using a larger number of components it is a good practice.
Finally, Fig. [[2.c]](#fig:1d_dataset) demonstrates how the learned model is able to map a dataset coming from an unknown density to the Beta distributin over-defined.


<div id="fig:1d_dataset">
    <table>
    <tr>
        <td style="text-align: center">
            <figure style="margin: 0px;">
            <img src="{{site.baseurl}}/assets/img/norm_flow/1d/dataset.png" style="max-width: 380px">
            <figcaption style="font-size:small;">
                Figure 2.a: Training dataset build by sampling 750 elements from two distinct gaussian distributions.
            </figcaption>
            </figure>
        </td>
        <td style="text-align: center">
            <figure style="margin: 0px;">
            <img src="{{site.baseurl}}/assets/img/norm_flow/1d/fit-model.png" style="max-width: 500px">
            <figcaption style="font-size:small;">
                Figure 2.b: A normalizing flow fitted to the given dataset to learn $p(x)$. The normalizing flow is composed by a beta distributon as a prior and as a gaussian mixture model with 4 different component as a flow.
            </figcaption>
            </figure>
        </td>
    </tr>
    <tr>
        <td colspan="2" style="text-align: center">
            <figure style="margin: 0px;">
            <img src="{{site.baseurl}}/assets/img/norm_flow/1d/learned-transformation.png">
            <figcaption style="font-size:small;">
                Figure 2.c: Learned normalizing flow from the unknown distribution $p(x)$ to the choosen prior $p(z)$.
            </figcaption>
            </figure>
        </td>
    </tr>
    </table>
</div>

Full code is contained in the following [notebook](https://github.com/andompesta/pytorch-normalizing-flows/blob/main/nf_demo.ipynb).

<!-- 2d example -->

<!-- review of different methods to achieve fast determinant computation -->



# Credits

The content of this post is based on the lectures and code of [Pieter Abbeel](https://sites.google.com/view/berkeley-cs294-158-sp20/home), [Justin Solomon](https://groups.csail.mit.edu/gdpgroup/6838_spring_2021.html) and [Karpathy's](https://github.com/karpathy/pytorch-normalizing-flows) tutorial.
Moreover, I want to credit [Lil'Long](https://lilianweng.github.io/posts/2018-10-13-flow-models/) and [Eric Jang](https://blog.evjang.com/2018/01/nf1.html) for their amazing tutorials. For example, the pioneering work done by [Dinh et. al.](#ref:nice) is the first to leverage transformations with triangular matrix for efficent determinatnt computation.

# Refences

<ol>
    <li id="ref:normalization-flow-review"> Papamakarios, G., Nalisnick, E., Rezende, D. J., Mohamed, S., & Lakshminarayanan, B. (2019). Normalizing Flows for Probabilistic Modeling and Inference. <a href="http://arxiv.org/abs/1912.02762">arxiv.org/abs/1912.02762</a></li>
    <li id="ref:change-of-variable"> Weisstein, Eric W. "Change of Variables Theorem." From MathWorld--A Wolfram Web Resource. <a href="https://mathworld.wolfram.com/ChangeofVariablesTheorem.html">mathworld.wolfram.com/ChangeofVariablesTheorem.html</a> </li>
    <li id="ref:density-estimation"> Dinh, L., Sohl-Dickstein, J., & Bengio, S. (2016). Density estimation using Real NVP. http://arxiv.org/abs/1605.08803</li>
    <li id="ref:ffjord"> Grathwohl, W., Chen, R. T. Q., Bettencourt, J., Sutskever, I., & Duvenaud, D. (2018). FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models. <a href="http://arxiv.org/abs/1810.01367">arxiv.org/abs/1810.01367</a> </li>
    <li id="ref:nf-anomaly-detection"> Hirschorn, O., & Avidan, S. (n.d.). Normalizing Flows for Human Pose Anomaly Detection. https://github.com/orhir/STG-NF. </li>
    <li id="ref:nf-for-odd"> Kirichenko, P., Izmailov, P., & Wilson, A. G. (n.d.). Why Normalizing Flows Fail to Detect Out-of-Distribution Data. https://github.com/PolinaKirichenko/flows_ood. </li>
    <li id="ref:glow"> Kingma, Durk P., and Prafulla Dhariwal. "Glow: Generative flow with invertible 1x1 convolutions." Advances in neural information processing systems 31 (2018). </li>
    <li id="ref:nice"> Dinh, L., Krueger, D., & Bengio, Y. (2014). NICE: Non-linear Independent Components Estimation. <a href="http://arxiv.org/abs/1410.8516"> arxiv.org/abs/1410.8516 </a> </li>
    <li id="ref:wave-net"> van den Oord, A., Dieleman, S., Zen, H., Simonyan, K., Vinyals, O., Graves, A., Kalchbrenner, N., Senior, A., & Kavukcuoglu, K. (n.d.). WAVENET: A GENERATIVE MODEL FOR RAW AUDIO. </li>
    <li id="ref:flow-wave-net"> Kim, Sungwon, et al. "FloWaveNet: A generative flow for raw audio." arXiv preprint arXiv:1811.02155 (2018). </li>
</ol>