---
layout: post
mathjax: true
title:  "Basic Principles of Linear Algebra"
author: "Sandro Cavallari"
tag: "Linear Algebra"
---

# Basis Vectors

In linear algebra, a vector basis $$B$$ of a vector space $$V$$ is a set of vectors $$\{\mathbf{b_1}, ..., \mathbf{b_n}\}$$ that are linearly independent and allow to reconstruct every vector $\mathbf{v_i} \in V$ as a linear combination of $B$:

$$
\begin{align*}
	\mathbf{v_i} & = a_1 \mathbf{b_1} + ... + a_n \mathbf{b_n} 
\end{align*}
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
\begin{align*}
	L(a_1 \mathbf{i} + a_2 \mathbf{j}) & = a_1L(\mathbf{i}) + a_2L(\mathbf{j})
\end{align*}
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

Note that as matrix multiplication is equal to applying different linear transformations, the multiplication order does matter.

# Determinant
As linear transformations alter the original vector space, it is important to evaluate by how much the original space is expanded or contracted by a given linear transformation $L$. The **determinant** define how much the original unit surface is changed by $$L$$.

The determinant has some interesting properties:
1. A liner transantformation with 0 determinant ($$ det(L) = 0$$) means that squash all the vectrs on a single line/plane. Moreover, it also means that $$L$$ has linearly dependents columns.
2. The determinant can be negative if it change orientation of the space.
3. Determinant is associative: $$det(L) \cdot det(M) = det(L \cdot M)$$.