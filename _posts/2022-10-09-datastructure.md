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


<div style="text-align:center;">
<table style="border:none; background:transparent; text-align:center;">
    <tbody>
    <tr>
        <th>Notation</th>
        <th>Name</th>
        <th>Example</th>
    </tr>
    <tr>
        <td>$ O(1) $</td>
        <td><a href="https://en.wikipedia.org/wiki/Time_complexity#Constant_time" class="mw-redirect" title="Constant time">constant</a></td>
        <td>Determining if a binary number is even or odd; <span> Using a constant-size <a href="https://en.wikipedia.org/wiki/Lookup_table" title="Lookup table">lookup table</a> </span>
        </td>
    </tr>
    <tr>
        <td>$O(\log N)$</td>
        <td><a href="https://en.wikipedia.org/wiki/Logarithmic_time" class="mw-redirect" title="Logarithmic time">logarithmic</a></td>
        <td>Finding an item in a sorted array with a binary search or a balanced search tree as well as all operations in a binomial heap.
        </td>
    </tr>
    <tr>
        <td> $O(N)$ </td>
        <td><a href="https://en.wikipedia.org/wiki/Linear_time" class="mw-redirect" title="Linear time">linear</a></td>
        <td>Finding an item in an unsorted list or in an unsorted array; adding two <i>n</i>-bit integers by ripple carry</td>
    </tr>
    <tr>
        <td>$ O(N\log N)=O(\log N!) $ </td>
        <td><a href="https://en.wikipedia.org/wiki/Linearithmic_time" class="mw-redirect" title="Linearithmic time">linearithmic</a>
        </td>
        <td>Performing a <a href="https://en.wikipedia.org/wiki/Fast_Fourier_transform" title="Fast Fourier transform">fast Fourier transform</a>; <a href="https://en.wikipedia.org/wiki/Heapsort" title="Heapsort">heapsort</a> and <a href="https://en.wikipedia.org/wiki/Merge_sort" title="Merge sort">merge sort</a>
        </td>
    </tr>
    <tr>
        <td> $O(N^{2}) $</td>
        <td><a href="https://en.wikipedia.org/wiki/Quadratic_time" class="mw-redirect" title="Quadratic time">quadratic</a></td>
        <td>Simple sorting algorithms, such as <a href="https://en.wikipedia.org/wiki/Bubble_sort" title="Bubble sort">bubble sort</a>, <a href="https://en.wikipedia.org/wiki/Selection_sort" title="Selection sort">selection sort</a> and <a href="https://en.wikipedia.org/wiki/Insertion_sort" title="Insertion sort">insertion sort</a>
        </td>
    </tr>
    <tr>
        <td>$ O(N!) $</td>
        <td><a href="https://en.wikipedia.org/wiki/Factorial" title="Factorial">factorial</a></td>
        <td>Solving the <a href="https://en.wikipedia.org/wiki/Travelling_salesman_problem" title="Travelling salesman problem">travelling salesman problem</a> via brute-force search; generating all unrestricted permutations of a <a href="https://en.wikipedia.org/wiki/Partially_ordered_set" title="Partially ordered set">poset</a>
        </td>
    </tr>
    </tbody>
</table>
</div>