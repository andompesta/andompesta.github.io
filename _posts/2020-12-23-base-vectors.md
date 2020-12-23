---
layout: post
mathjax: true
title:  "Basic Principles of Linear Algebra"
author: "Sandro Cavallari"
tag: "Linear Algebra"
---

# Basis Vectors

In linear algebra, a vector basis $$B$$ of a vector space $$V$$ is a set of vectors $$\{\mathbf{b_1}, ..., \mathbf{b_n}\}$$ that are linearly independet and allow to reconstruct every vector $\mathbf{v_i} \in V$ as linear combination of $B$:
$$
\begin{align}
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
\right]$$ are the most common base vectors for the vectors space in $$\mathbb{R}^2$$. Thus, it is possible to represet a vector $$ \mathbf{x} = \left[\begin{array}{c}
  3\\
  2
\end{array}\right]$$ as $$ 3 \mathbf{i} + 2 \mathbf{j}$$.

<div>
<img src="{{site.baseurl}}/assets/img/linear_algebra/basic_vectors.png" alt="Base Vectors">
<p style="font-size:small;">Figure 1: Vector representation by base vector.</p>
</div>


The ability of represent any vector in $$V$$ as linear combination of the basis vectors is a powerful concept.
However, $$\mathbf{i}$$ and $$\mathbf{j}$$ are not the only possible basis vectors of $$\mathbb{R}^2$$.
For example, an other possible basis is  $$ \mathbf{v} = \left[\begin{array}{c}
  1\\
  2
\end{array}\right]$$ and  $$ \mathbf{w} = \left[\begin{array}{c}
  3\\
  -1
\end{array}\right]$$. However, the representation of $$\mathbf{x}$$ w.r.t. $$\mathbf{v}$$ and $$\mathbf{w}$$ would be different than the one w.r.t. $$\mathbf{i}$$ and $$\mathbf{j}$$.


# Span

The **Span** is defined as the set of all possible vectors that we can create given a basis set.
Note that, the number of basis vectors define the dimention of our vector space.

# Linear Transformations

A **linear transformation** is equvalent to a function over vectors. That is, a linear transformation "move" an input vector to an output vector. While general transformations have complex features, linear transformations have some well defined properties:
1. they mantains the origin of the vector space invariante
2. they maps equally spaced lines to equally spaced lines (or points)

$$
\begin{align}
	L(a_1 \mathbf{i} + a_2 \mathbf{j}) & = a_1L(\mathbf{i}) + a_2L(\mathbf{j})
\end{align}
$$

<div>
<img src="{{site.baseurl}}/assets/img/linear_algebra/transformations.png" alt="Base Vectors" style="max-width: 70%">
<p style="font-size:small;">Figure 2: Examples of the most commont linear transformations. (Image taken from <a href="https://mathigon.org/course/linear-algebra/linear-transformations"> Samuel S. Watson</a>)</p>
</div>


Thanks to their properties, it is possible to linearly transforma any vector by means to its basis. 
In other words, given a vector $$\mathbf{x} = \left[\begin{array}{c} -1\\ 2 \end{array}\right]$$ w.r.t. $$\mathbf{i}$$ and $$\mathbf{j}$$ and any lineart transformation $$L$$.
It is possible to describe $$L(\mathbf{x}) = \left[\begin{array}{c} -1\\ 2 \end{array}\right] $$ w.r.t. $$L(\mathbf{i})$$ and $$L(\mathbf{j})$$ (formally $$L(\mathbf{x}) = -1 L(\mathbf{i}) + 2 L(\mathbf{j})$$).

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


Finally, as a linear transformation is represented by a matrix, it is possible to define the **composition of two or more linear transformations** as he left-to-right product of the transformation matrixis:

$$
\begin{align*}
L_2(L_1( \mathbf{x} )) =  L_2L_1(\mathbf{x})
\end{align*}
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

