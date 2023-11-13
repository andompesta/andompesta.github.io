---
layout: post
mathjax: true
title:  "Basic Principles of Linear Algebra"
author: "Sandro Cavallari"
tag: "Linear Algebra"
comments_id: 2
---


Linear algebra is the branch of math and statistics that is devoted to the study of matrices and vectors. As such, it is broadly used to model real-world problems in phisitcs and machine learning. Such post is a collections of my notes obtained from the 3Blue1Brown series on linear-algebra <a href="#ref:3b1b">[1]</a> and Murphy's new book <a href="#ref:mkp">[2]</a>.

# Basic Matrix Operations

1. **transpose**: given a matrix $$\rmA \in \mathbf{R}^{m \times n}$$, its transpose $$\rmA^T$$ is obtained ''flipping'' the rows and colums

$$
\rmA = \left[\begin{array}{cccc}
  a_{11} & a_{12} & ... & a_{1n}\\
  a_{21} & a_{22} & ... & a_{2n}\\
  \vdots & \vdots & \vdots & \vdots \\
  a_{m1} & a_{m2} & ... & a_{mn}\\
\end{array}\right]
\Rightarrow
\rmA^T = \left[\begin{array}{cccc}
  a_{11} & a_{21} & ... & a_{m1}\\
  a_{12} & a_{22} & ... & a_{m2}\\
  \vdots & \vdots & \vdots & \vdots \\
  a_{1n} & a_{2n} & ... & a_{mn}\\
\end{array}\right].
$$

The most important properties are:
* $${(\rmA^T)}^T = \rmA$$
* $$(\rmA \rmB)^T = \rmB^T \rmA^T$$
* $$(\rmA + \rmB)^T = \rmA^T \rmB^T$$


{:start="2"}
2. **matrix multiplication**: while the summation of 2 matrixes is done element-wise. Matrix multiplication is done row-by-colum and requires matrixes of specific sizes. Given $$A \in \mathbf{R}^{m \times n}$$ and $$B \in \mathbf{R}^{n \times p}$$ it is possible to define $$\rmC = \rmA \rmB \in \mathbf{R}^{m \times p}$$ s.t. $$c_{i,j} = \sum_{k=1}^{n} a_{ik} b_{kj}$$. In other words, $$\rmC$$ is a linear combination of the row of $$\rmA$$ and the colum of $$\rmB$$.

$$
\rmC = \left[\begin{array}{ccc}
  - & \rva_{1:} & -\\
  - & \rva_{2:} & -\\
     & \vdots &   \\
  - & \rva_{m:} & -\\
\end{array}\right]
\left[\begin{array}{cccc}
  | & | & | & |\\
  \rvb_{:1} & \rvb_{:2} & \dots & \rvb_{:p}\\
  | & | & | & |\\
\end{array}\right]
= 
\left[\begin{array}{cccc}
  \rva_{1:}\rvb_{:1} & \rva_{1:}\rvb_{:2} & \dots & \rva_{1:}\rvb_{:p}\\
  \rva_{2:}\rvb_{:1} & \rva_{2:}\rvb_{:2} & \dots & \rva_{2:}\rvb_{:p}\\
     & \vdots & \vdots &   \\
  \rva_{m:}\rvb_{:1} & \rva_{m:}\rvb_{:2} & \dots & \rva_{m:}\rvb_{:p}\\
\end{array}\right].
$$

The most important properties are:
* $$(\rmA \rmB) \rmC = \rmA (\rmB \rmC)$$
* $$\rmA(\rmB + \rmC) = \rmA \rmB \ \rmA \rmC)$$
* $$\rmA \rmB \neq \rmB \rmA $$

{:start="3"}
3. **matrix inverse**: As for real numbers,  the inverso of a matrix $$\rmA$$ is denoted as $$\rmA^{-1}$$ and is defined as the matrix such that: $$\rmA \rmA^{-1} = \rmI$$. Besites being easy to define computing the inverse of a matrix is an expencive operations. Moreover, $$\rmA^{-1}$$ exists if and only if $$det(\rmA) \neq 0$$.

The most important properties are:
* $$(\rmA^{-1})^{-1} = \rmA$$
* $$(\rmA \rmB)^{-1} = \rmB^{-1} \rmA^{-1}$$
* $$(\rmA^{-1})^{T} = (\rmA^{T})^{-1} = \rmA^{-T}$$

# Basis Vectors

In linear algebra, a vector basis $$\rmB$$ of a vector space $$\rmV$$ is a set of vectors $$\{\rvb_1, ..., \rvb_n\}$$ that are linearly independent and allow to reconstruct every vector $$\mathbf{v_i} \in V$$ as a linear combination of $$\rmB$$:

$$
\begin{align}
\label{eq:basis_vector}
	\mathbf{v_i} & = a_1 \mathbf{b_1} + ... + a_n \mathbf{b_n} 
\end{align}
$$


For example, the vectors 
$$ \mathbf{i} = \left[
\begin{array}{c}
  1\\
  0
\end{array}
\right] $$
and $$\mathbf{j} = \left[
\begin{array}{c}
  0\\
  1
\end{array}
\right]$$ are the most common base vectors for the vectors space $$\mathbb{R}^2$$. Thus, it is possible to represent a vector $$ \mathbf{x} = \left[\begin{array}{c}
  3\\
  2
\end{array}\right]$$ as $$ 3 \mathbf{i} + 2 \mathbf{j}$$.

<div>
<img src="{{site.baseurl}}/assets/img/linear_algebra/basic_vectors.png" alt="Base Vectors">
<p style="font-size:small;">Figure 1: Vector representation by the base vector $\mathbf{i}$ and $\mathbf{j}$.</p>
</div>


The ability to represent any vector in $$V$$ as a linear combination of the basis vectors is a powerful concept.
However, $$\mathbf{i}$$ and $$\mathbf{j}$$ are not the only possible basis vectors of $$\mathbb{R}^2$$.
For example, another possible basis could be formed by $$\mathbf{v} = \left[\begin{array}{c}
  1\\
  2
\end{array}\right]$$ and  $$ \mathbf{w} = \left[\begin{array}{c}
  3\\
  -1
\end{array}\right]$$. However, the representation of $$\mathbf{x}$$ w.r.t. $$\mathbf{v}$$ and $$\mathbf{w}$$ would be different than the one w.r.t. $$\mathbf{i}$$ and $$\mathbf{j}$$.


# Span

The **Span** is defined as the set of all possible vectors that we can create given a basis set.
Note that the number of basis vectors defines the dimension of our vector space.

# Linear Transformations

A **linear transformation** is equivalent to a function over vectors. That is, a linear transformation "move" an input vector to an output vector. While general transformations have complex features, linear transformations have some well-defined properties:
1. they maintain the origin of the vector space invariant
2. they map equally spaced lines to equally spaced lines (or points)

$$
\begin{align}
\label{eq:linear_transform}
	L(a_1 \mathbf{i} + a_2 \mathbf{j}) & = a_1L(\mathbf{i}) + a_2L(\mathbf{j})
\end{align}
$$

<div>
<img src="{{site.baseurl}}/assets/img/linear_algebra/transformations.png" alt="Base Vectors" style="max-width: 70%">
<p style="font-size:small;">Figure 2: Examples of the most commont linear transformations. (Image taken from <a href="https://mathigon.org/course/linear-algebra/linear-transformations"> Samuel S. Watson</a>)</p>
</div>

Thanks to their properties, it is possible to linearly transform any vector by means to its basis. In other words, given a vector $$\mathbf{x} = \left[\begin{array}{c} -1\\ 2 \end{array}\right]$$ w.r.t. $$\mathbf{i}$$ and $$\mathbf{j}$$ and any lineart transformation $$L$$.
It is possible to represent $$L(\mathbf{x})$$ as a function of $$L(\mathbf{i})$$ and $$L(\mathbf{j})$$ (formally $$L(\mathbf{x}) = -1 L(\mathbf{i}) + 2 L(\mathbf{j})$$).

For example, assume $$L = \left[\begin{array}{cc} 1 & 3\\ -2 & 0 \end{array}\right]$$, then:

$$
\begin{align*}
L(\mathbf{i}) &= \left[\begin{array}{cc} 1 & 3\\ -2 & 0 \end{array}\right]  \left[\begin{array}{c} 1\\ 0 \end{array}\right] = \left[\begin{array}{c} 1\\ -2 \end{array}\right]\\
L(\mathbf{j}) &= \left[\begin{array}{cc} 1 & 3\\ -2 & 0 \end{array}\right]  \left[\begin{array}{c} 0\\ 1 \end{array}\right] = \left[\begin{array}{c} 3\\ 0 \end{array}\right] \\
L(\mathbf{x}) &= -1 L(\mathbf{i}) + 2 L(\mathbf{j}) \\
	&= -1 \left[\begin{array}{c} 1\\ -2 \end{array}\right] + 2 \left[\begin{array}{c} 3\\ 0 \end{array}\right] \\
	&= \left[\begin{array}{c} 5\\ 2 \end{array}\right]
\end{align*}
$$

<div>
<img src="{{site.baseurl}}/assets/img/linear_algebra/linear-transform.png" style="max-width: 85%">
<p style="font-size:small;">Figure 3: Visualization of the linear transformation appled to vector x.)</p>
</div>



Finally, as a linear transformation is represented by a matrix, it is possible to define the **composition of two or more linear transformations** as he left-to-right product of the transformation matrix:

$$
\begin{align}
L_2(L_1( \mathbf{x} )) =  L_2L_1(\mathbf{x})
\end{align}
$$

For example, if $$L_1 = \left[\begin{array}{cc} 1 & -2\\ 1 & 0 \end{array}\right]$$, $$L_2 = \left[\begin{array}{cc} 0 & 2\\ 1 & 0 \end{array}\right]$$ and $$\mathbf{x} = \left[\begin{array}{c} x\\ y \end{array}\right]$$. Then:


$$
\begin{align*}
L_2(L_1( \mathbf{x} )) &= \left[\begin{array}{cc} 0 & 2\\ 1 & 0 \end{array}\right] \Big ( \left[\begin{array}{cc} 1 & -2\\ 1 & 0 \end{array}\right] \left[\begin{array}{c} x\\ y \end{array}\right] \Big)
\\
L_2 L_1( \mathbf{x} ) &= \Big ( \left[\begin{array}{cc} 0 & 2\\ 1 & 0 \end{array}\right] \left[\begin{array}{cc} 1 & -2\\ 1 & 0 \end{array}\right] \Big) \left[\begin{array}{c} x\\ y \end{array}\right] \\
&= \left[\begin{array}{cc} 2 & 0\\ 1 & -2 \end{array}\right] \left[\begin{array}{c} x\\ y \end{array}\right] 
\end{align*} 
$$

Note that as matrix multiplication is equal to applying different linear transformations, the multiplication order does matter.

# Determinant
As linear transformations alter the original vector space, it is important to evaluate by how much the original space is expanded or contracted by a given linear transformation $L$. The **determinant** define how much the original unit surface is changed by $$L$$.

The determinant has some interesting properties:
1. A liner transantformation with 0 determinant ($$ det(L) = 0$$) means that squash all the vectrs on a single line/plane. Moreover, it also means that $$L$$ has linearly dependents columns.
2. The determinant can be negative if it change orientation of the space.
3. Determinant is associative: $$det(L) \cdot det(M) = det(L \cdot M)$$.

<div>
<img src="{{site.baseurl}}/assets/img/linear_algebra/determinant.png" style="max-width: 85%">
<p style="font-size:small;">Figure 4: Visualization of the determinant for a initial vector space defined by $\mathbf{i}$ and $\mathbf{j}$ and the vector space obtained after applying the transformation $L$.)</p>
</div>


# System of Linear Equations
It is convininet to use inear algebra to represent a system of linear equations, e.g.


<table align="center">
<tr>
  <td>
    $$\begin{cases} 
    2x + 5y + 3z = -3 \\ 
    4x + 8z = 0 \\
    x + 3y = 2
    \end{cases}$$
  </td>
  <td>
    $$
    \Rightarrow
    $$
  </td>
  <td style="width: 5px">
    $$
    \left[\begin{array}{ccc} 
    2 & 5 & 3\\
    4 & 0 & -2\\
    1 & 3 & 0
    \end{array}
    \right]
    $$
  </td>
  <td style="width: 5px">
    $$
    \left[\begin{array}{c} 
    x\\
    y\\
    z
    \end{array}
    \right]
    $$
  </td>
  <td style="width: 3px">
    = 
  </td>
  <td style="width: 5px">
    $$
    \left[\begin{array}{c} 
    -3\\
    0\\
    2
    \end{array}
    \right]
    $$
  </td>
</tr>
<tr>
    <td></td>
    <td></td>
    <td>$$\rmA$$</td>
    <td>$$\rvx$$</td>
    <td> = </td>
    <td>$$\rvb$$</td>
</tr>
</table>

Thus, any system of linear equations can be expressed as:

$$
\begin{equation}
\rmA \rvx = \rvb
\end{equation}
$$

where $$\rmA \in \mathbb{R}^{m \times n}$$ is a known linear transformation(a matrix), $$\rvb \in \mathbb{R}^{m \times 1}$$ is a known vector in the space of $$\rmA$$, and $$\rvx \in \mathbb{R}^{n \times 1}$$ is an unkown vector that after the transformation $$\rmA$$ lies over $$\rvb$$.


Note that the existence of such unkown vector is tightly related to the determinant of $$\rmA$$:
* if $$det(A) = 0$$, in general, there is no such $$\rvx$$
* if $$det(A) \neq 0$$, in general, there is one-and-only-one $$\rvx$$ that satisfy $$ \rmA \rvx = \rvb $$, namely $$\rmA^{-1}$$.

Mathematically, the solution to $$\rmA \rvx = \rvb$$ is $$\rvx = \rmA^{-1} \rvb$$. However, computing $$\rmA^{-1}$$ is a complex operation and is subject to numerical instabilities ([Don’t invert that matrix](https://www.johndcook.com/blog/2010/01/19/dont-invert-that-matrix/)). Thus, mathematicians have develop multiple solvers for that same problam that does not require to compute the matrix invers and they leverage some specific property of matrix $$\rmA$$.


# Change of Basis

Given a vector $$\rvx = \left[\begin{array}{c} 3\\ 2\end{array}\right]$$ imagine this vector represented in terms of the unit vectors $$\rvi = \left[\begin{array}{c} 1\\ 0\end{array}\right]$$ and $$\rvj = \left[\begin{array}{c} 0\\ 1\end{array}\right]$$, and, scale them by 3 and 2, i.e.


$$\rvx = \left[\begin{array}{c} 3 \rvi\\ 2 \rvj \end{array}\right]$$

However, as shown if Fig. 5, we can also represent $$\rvx$$ in terms of different basis vectors $$\rvu = \left[\begin{array}{c} 2\\ 1\end{array}\right]$$ and $$\rvv = \left[\begin{array}{c} -1\\ 1\end{array}\right]$$. That is, $$\rvx$$ can be represented as the linear combination of $$\rvu$$ $$\rvj$$:

$$\rvx = \left[\begin{array}{c} \frac{5}{3} \rvu\\ \frac{1}{3} \rvv \end{array}\right]$$

In other words, it is possible to represent $$\rvx$$ in two different languages: one according to basis $$\rvi$$ $$\rvj$$ the other according to basis $$\rvu \rvv$$.

<div>
<img src="{{site.baseurl}}/assets/img/linear_algebra/base_change.png" style="max-width: 85%">
<p style="font-size:small;">Figure 5: Visualization of the same vector $\rvx$ represented according to two different basis vectors.)</p>
</div>


As overstated, we can express $$\rvu$$ and $$\rvv$$ in terms of basis vectors $$\rvi$$ and $$\rvj$$ as:

$$\rvu = \left[\begin{array}{c} 2\\ 1\end{array}\right] ~~~~~~ \rvv = \left[\begin{array}{c} -1\\ 1\end{array}\right]$$

or in terms of $$\rvu$$ $$\rvv$$ it-self:

$$\rvu = \left[\begin{array}{c} 1\\ 0\end{array}\right] ~~~~~~ \rvv = \left[\begin{array}{c} 0\\ 1\end{array}\right]$$


Yet, the linear transformation $$\rmU = [\rvu, \rvv]$$ (composed by the collum vectors $$\rvu$$ and $$\rvv$$) allow to convert any vector written in terms of $$\rvu$$ $$\rvv$$ to its equivalent vector w.r.t. $$\rvi$$ and $$\rvj$$:

$$ \left[\begin{array}{cc} 2 & -1\\ 1 & 1\end{array}\right] \cdot \left[\begin{array}{c} \frac{5}{3}\\ \frac{1}{3}\end{array}\right] = \left[\begin{array}{c} 3\\ 2 \end{array}\right]$$

Similarly, we can use $$\rmU^{-1}$$ to convert any vector written in terms of $$\rvi$$ $$\rvj$$ to it equivalent representeation in $$\rvu$$ $$\rvv$$:

$$
\left[\begin{array}{cc} \frac{1}{3} & \frac{1}{3}\\ -\frac{1}{3} & \frac{2}{3}\end{array}\right] \cdot \left[\begin{array}{c} 3\\ 2\end{array}\right] = \left[\begin{array}{c} \frac{5}{3}\\ \frac{1}{3}\end{array}\right]
$$


More generaly, any transformation $$\rmA$$ expressed in terms of the basis $$\rvi$$ and $$\rvj$$ can be applyed to any vectror $$\rvx$$ defined in temrs of basis $$\rvu$$ and $$\rvv$$ applying the change-of-basis equation:

$$
\begin{equation}
\label{eq:cob}
[\rmU^{-1} \rmA \rmU] \rvx
\end{equation}
$$

where $$\rmU^{-1} \rmA \rmU$$ express a sort of mathematical empathy between different reference systems; i.e., it converts a tranformation $$\rmA$$ to a different reference systems.

# Eigenvectors and Eigenvalues

It is often convinient to study linear transformations, not on their matrix formulation, but ratehr on their base component.
Among the most common decomposition methods, **eigenvectors** and **eigenvalues** are the most common matrix decomposition thecnique.


Given a linear transformation $$L = \left[\begin{array}{cc} 3 & 1\\ 0 & 2 \end{array}\right]$$, most of the vectors $$\mathbf{v}_i$$ are rotated by $$L$$ away from their original span.
Instead some special vectors $$\mathbf{e}_i$$ are only streched or squished by $$L$$, but they remain on the original span. Moreover, every vector on the span of $$\mathbf{e}_i$$ is also only scaled by $$L$$.


<div>
<img src="{{site.baseurl}}/assets/img/linear_algebra/eigen_values.png" style="max-width: 85%">
<p style="font-size:small;">Figure 6: Visualization of one of the eigenvalue of $L$. Note that, $\mathbf{e}_1$ remain on its own span, while a random vector $\mathbf{v}$ is moved away from its original span.)</p>
</div>


Base on the intuition shown in Fig. 6 and on the "move from the span" consepts, we can formally define the eigenvalues of a squared matrix $$\rmA \in \mathbb{R}^{n \times n}$$ as the non-zero vector $$\mathbf{e}_i$$:


$$
\begin{align}
\rmA \cdot \mathbf{e}_i = \lambda \mathbf{e}_i
\end{align}
$$

where:
* $$\lambda$$ is known as the eigenvalue of the eigenvector $$\mathbf{e}$$.
* $$\lambda \neq 0 $$.
* if $$\mathbf{e}_i$$ is an eigenvectors of $$\rmA$$, then any rescaled vector $$c ~ \mathbf{e}_i $$ for $$ c \in \mathbb{R}, c \neq 0$$ is also an eigenvectors of $$\rmA$$. Thus, usually only the unit eigenvectors are considered.

There is an interesting connection between eigenvectors are determinant. According to the formal definition of eigenvectors, we are tring to map a matrix to a vector. Thus, we are tring to map a volume/surface to a single line/point; which is possible only if the determinant of the matrix is 0:

$$
\begin{align}
\label{eq:eigenvectors}
\rmA \cdot \mathbf{e}_i &= \lambda \mathbf{e}_i \\
\rmA \cdot \mathbf{e}_i &= (\rmI \lambda) \mathbf{e}_i \nonumber \\
(\rmA - \lambda \rmI) \mathbf{e}_i &= 0 \nonumber \\
& \Rightarrow det(\rmA - \lambda \rmI) = 0 \nonumber
\end{align}
$$


The most important properties are:
* the trance of a matrix is equal to the some of its eigenvalues: $$tr(\rmA) = \sum_{i=0}^{n-1} \lambda_i$$
* the determinanto of $$\rmA$$ is equal to the producto of its eigenvalues: $$ det(\rmA) = \prod_{i=0}^{n-1} \lambda_i$$



# Matrix Decomposition

Similarly, to how it is conveninet to express $$15$$ as product of its factors $$5 \cdot 3$$; sometimes it is convenient to express a matrix $$\rmA$$ as product of other matrixes.
There are multiple method to decpompose a matrix, but they are mostly used to eficiently solve systems of linear equation.

## Eigendecomposition 

Given a squared matrix $$\rmA \in \mathbb{R}^{n \times n}$$, it is possible to rewrite Eq. \ref{eq:eigenvectors} in matrix form as:

$$
\begin{equation}
\label{eq:eigenvectors_matrix}
\rmA \rmU = \rmU \mathbf{\Lambda}
\end{equation}
$$

Moreover, according to Eq. \ref{eq:cob}, using the eigenvectors of $$\rmA$$ as new basis of $$\rmA$$ will generate a diagonal matrix of eigenvalues:

$$
\begin{equation}
\rmU^{-1} \rmA \rmU = \mathbf{\Lambda}
\end{equation}
$$

where $$\rmU \in \mathbb{R}^{n \times n} = \left[\begin{array}{ccc}
  | & | & | \\
  \rve_1 & \dots & \rve_{n}\\
  | & | & | \\
\end{array}\right]$$ is the matrix formed by the eigenvectors of $$\rmA$$ and $$\mathbf{\Lambda} \in \mathbb{R}^{n \times n} = \left[\begin{array}{ccc}
  \lambda_1 &  &  \\
   & \ddots & \\
   & & \lambda_n \\
\end{array}\right]$$ is the diagonal matrix formed by the eigenvalues assogiated to the eigenvectors of $$\rmA$$.


This process of expressing $$\rmA$$ in terms of its eigenvalue and eigenvectors is know as **diagonalization**.
If the eigenvalues of $$\rmA$$ are linearly indipendent, then the matrix $$\rmU$$ is invertible, thus, it is possible to **decompose** $$\rmA$$ as:

$$
\begin{equation}
\label{eq:eigendecomposition}
\rmA = \rmU \mathbf{\Lambda} \rmU^{-1} .
\end{equation}
$$

Moreover, if $$\rmA$$ is real valued and symmetric then it can be shown that $$\rmU$$ is orthonormal, i.e., $$\rvu^T_i \rvu_j = 0$$ if $$i \neq j$$ and $$\rvu^T_i \rvu_i = 1$$ (or $$\rmU^T\rmU = \rmU \rmU^T = \rmI$$). Thus, we can futher symplify Eq. \ref{eq:eigendecomposition} $$\rmA$$ as:

$$
\begin{equation}
\rmA = \rmU \mathbf{\Lambda} \rmU^T .
\end{equation}
$$

As a final note, it is possible to leverage such eigendecomposition to easily compute the inverse of a matrix $$\rmA$$. Since $$\rmU^T = \rmU^{-1}$$, we have:

$$
\begin{equation}
\rmA^{-1} = \rmU \mathbf{\Lambda}^{-1} \rmU^T .
\end{equation}
$$


### Lagrangian Methods for Constrained Optimization
While eigen decomposition is commonly applied to solve systems of liear equations.
It is also a powerful method for optimization subject to linear constrains (constrained optimization).
That is, it can be used to solve quadratic constrained problems of the form:

$$
\min_{\rvx} \rvx^T \rmH \rvx + d, ~~\text{subject to} ~~ \rvx^T \rvx - 1 = 0
$$

where $$\rmH \in \mathbb{R}^{n \times n}$$ is symmetric.
Such problems are a specific instanche of the **Lagrangian method**, in which an augmented objective is created to ensure the constrain satisfability:

$$
L(\rvx, \lambda) = \max_{\lambda} \min_{\rvx} \rvx^T \rmH \rvx + d - \lambda (\rvx^T \rvx - 1)
$$

The optimal $$\rvx^*$$ that solve the problem, need to satisfy the zero-gradient condition:

$$
\begin{align*}
\frac{\partial L(\rvx, \lambda)} {\partial \rvx} & = 0 \\
 & = \frac{ \partial } {\partial \rvx} \rvx^T \rmH \rvx   +  \frac{\partial}{\partial \rvx} d - \frac{\partial}{\partial \rvx} \lambda (\rvx^T \rvx - 1)  \\
 & = \rvx^T (\rmH + \rmH^T) + 0 - 2 \lambda \rvx^T  && { \small \rmH = \rmH^T \text{ since is symmetric.} }\\
 & = 2 \rvx^T \rmH - 2 \lambda \rvx^T \\ 
\frac{\partial L(\rvx, \lambda)} {\partial \lambda} & = 0  \\
 & =  \frac{ \partial }{ \partial \lambda } \rvx^T \rmH \rvx + \frac{ \partial }{ \partial \lambda } d - \frac{ \partial }{ \partial \lambda } \lambda (\rvx^T \rvx - 1) \\
 & = 0 + 0 - \rvx^T \rvx + 1 \\
 & = \rvx^T \rvx - 1
\end{align*}
$$

which is equivalent to the eigenvector equation (Eq. \ref{eq:eigenvectors_matrix}) $$ \rmH \rvx = \lambda \rvx$$.




## Singular Value Decomposition (SVD)

While eigendecomposition require squared matrices, **SVD** allow the factorization of rectangular matrices into **singular vectors** and **singular values**. Given any $$\rmA \in \mathbb{R}^{m \times n}$$, it is possible to depompose it as:

$$
\begin{equation}
\rmA = \rmU \rmS \rmV^T
\end{equation}
$$

where $$\rmU \in \mathbb{R}^{m \times m}$$ is composed by orthonormal columns ($$\rmU^T \rmU = \rmI$$), $$\rmV \in \mathbb{R}^{n \times n}$$ is compesed by orthonormals rows and columns ($$\rmV^T\rmV = \rmV \rmV^T = \rmI$$), and $$\rmS \in \mathbb{R}^{m \times n}$$ is a diagonal matrix containing the **singular values** $$\sigma_i \geq 0$$.
$$\rmU$$ and $$\rmV^T$$ are respectively known as the **left singular vectors** and **right singular vectors** of $$\rmA$$ and are obtained as the eigenvectors of $$\rmA\rmA^T$$ and $$\rmA^T\rmA$$. Similarly, $$\rmS$$ is composed by the squared root of the eigenvalues of $$\rmA\rmA^T$$ and $$\rmA^T\rmA$$ arranged in descending order.

For example, consider

$$
\rmA = 
\left[\begin{array}{cc}
  2 & 4 \\
  1 & 3 \\
  0 & 0 \\
  0 & 0 \\
\end{array}\right]
$$

then we know that the columns of $$\rmU$$ are made by the eigenvalues of $$\rmA \rmA^T$$:

$$
\begin{align*}
\rmA \rmA^T &= \left[\begin{array}{cccc}
  20 & 14 & 0 & 0 \\
  14 & 10 & 0 & 0 \\
  0 & 0 & 0 & 0 \\
  0 & 0 & 0 & 0 \\
\end{array}\right]\\
\rmU &= \left[\begin{array}{cccc}
  0.82 & -0.58 & 0 & 0 \\
  0.58 & 0.82 & 0 & 0 \\
  0 & 0 & 1 & 0 \\
  0 & 0 & 0 & 1 \\
\end{array}\right]
\end{align*}
$$

similarly, the right singular vectors are obtained as eigenvalues of $$\rmA^T \rmA$$:

$$
\begin{align*}
\rmA^T \rmA &= \left[\begin{array}{cc}
  5 & 11 \\
  11 & 25\\
\end{array}\right]\\
\rmV &= \left[\begin{array}{cc}
  0.4 & -0.91 \\
  0.91 & 0.4
\end{array}\right]
\end{align*}
$$

instead, $$\rmS$$ is formed by the squared root of the eivenvectors of $$\rmV$$ or $$\rmU$$:

$$
\rmS = \left[\begin{array}{cc}
  5.46 & 0 \\
  0 & 0.37 \\
  0 & 0 \\
  0 & 0 
\end{array}\right]
$$


# Reference


<ol>
    <li id="ref:3b1b">3Blue1Brown. <a href="https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab"><b>Essence of Linear Algebra.</b></a></li>
    <li id="ref:mkp">Murphy, Kevin  Patrick. <b>Probabilistic Machine Learning: An Introduction.</b> MIT Press, 2021.</li>
</ol>
