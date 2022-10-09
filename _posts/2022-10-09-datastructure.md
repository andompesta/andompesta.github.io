---
layout: post
mathjax: true
title:  "Data structures"
author: "Sandro Cavallari"
tag: "Interviews"
---

Data scrtucures are efficent memory construct used to sotre and organize data in an efficent manner.
Adopting the right data structure and having efficent access to the needed information is a fundamentala to build usable and scalable products.


# Big-O Notation and asymptotic Analysis

To evaluate the efficency of a data structure we need to evaluate the **time** and **memory consumption** requred to execute the algorithm.
As the run-time depends on the input size, we will focus on the performance of the data structure when the inputs are infinitly large.
The asymptotic notations is the mathematical tool used to perform this analysis, specifically we will focus on the **Big-O** notation that studies the behaviout of each algorithm in the worst-case scenarious. Thus, it indicates the complexity of an algirithm assuming inputs of size N with $$\lim N\to\infty$$. Under this context constant factors are ignored as are dominated by N.

