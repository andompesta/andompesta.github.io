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


# Data Structures

## Array

Arrays are collections of items stored at a contiguous memory locations.
Such property makes array easy to traverse and genearlly it provides random access to its element in constant complexity.

Genearally speaking arrays have fixed size and new element can't be added if the array is already full.
However, it is possible to implement dynamic arrays at the expences of a memory overhead (unused memory is reserved for new items that will be added later on).
Dynamic arrays achieve constant time complexity when it comes to append and delete operation in the general case, but if resize is needed then a new copy of the current array has to be create; thus requireing high memory and time complexity.
The dynamic structure is obtained by creating a new array double size of the original array and copy all element from the previous array to the new array.

<div style="text-align:center;">
<table style="border:none; background:transparent; text-align:center;">
<tbody>
    <tr>
        <th>Name</th>
        <th>Description</th>
        <th>Complexity</th>
    </tr>
    <tr>
        <td> <b>Append</b> </td>
        <td> Add an element to the end of the array </td>
        <td> Time and Space: $O(1)$ (in ammortized time) </td>
    </tr>
    <tr>
        <td> <b>Insert</b> </td>
        <td> Insert an element to the i-th position of the array </td>
        <td> Time and Space: $O(N)$ </td>
    </tr>
    <tr>
        <td> <b>Remove</b> </td>
        <td> Remove the i-th element of the array </td>
        <td> Time: $O(N)$ and Space: $O(N)$ </td>
    </tr>
    <tr>
        <td> <b>Remove Last</b> </td>
        <td> Remove the last element of the array </td>
        <td> Time and Space: $O(1)$ (in ammortized time) </td>
    </tr>
    <tr>
        <td> <b>Search</b> </td>
        <td> Check if an element is present in the list </td>
        <td> Time: $O(N)$ and Space: $O(1)$ </td>
    </tr>
    <tr>
        <td> <b>Get</b> </td>
        <td> Get the i-th element in the list </td>
        <td> Time: $O(1)$ and Space: $O(1)$ </td>
    </tr>
    <tr>
        <td> <b>Sort</b> </td>
        <td> Get the i-th element in the list </td>
        <td> Time: $O(N \log N)$ and Space: $O(N)$ </td>
    </tr>
</tbody>
</table>
</div>


### Hash Tables
Hash tabels are one of the most importat data strcutre build uppon arrays. By organising data in (key, values) pairs it allows for fast insertion, lookup and access to data.
It is composed by an array and the position of each key in this array is determined by the function:

$$idx = hash(key) \% size(hash\_table)$$.

Python provide a native implementation of hash table under the dict class.

<div style="text-align:center;">
<table style="border:none; background:transparent; text-align:center;">
<tbody>
    <tr>
        <th>Name</th>
        <th>Description</th>
        <th>Complexity</th>
    </tr>
    <tr>
        <td> <b>Insert</b> </td>
        <td> Add an element to the dictionary </td>
        <td> Time and Space: $O(1)$ (in ammortized time) </td>
    </tr>
    <tr>
        <td> <b>Remove</b> </td>
        <td> Remove a key from the dictonary </td>
        <td> Time: $O(1)$ and Space: $O(1)$ </td>
    </tr>
    <tr>
        <td> <b>Search</b> </td>
        <td> Check if a key is present in the dictionary </td>
        <td> Time: $O(1)$ and Space: $O(1)$ </td>
    </tr>
    <tr>
        <td> <b>Get</b> </td>
        <td> Get a given key in the dictionary </td>
        <td> Time: $O(1)$ and Space: $O(1)$ </td>
    </tr>
    <tr>
        <td> <b>Iterate</b> </td>
        <td> Iterate over all element of the dictionary </td>
        <td> Time: $O(N)$ and Space: $O(1)$ </td>
    </tr>
</tbody>
</table>
</div>


## Linked List
A linked list is a linear data structure that includes a series of connected nodes.
Usually every nodes is composed by a data filed that contains some value and a pointer to the next element (if there is).
While arrays are contiguous in memory, linked lists allows for a dynamic memory management where nodes can be scattered across the memory and simply point to each other.
Linked lists are the fundamental backbone for other data structure as stacks and queue.

<div style="text-align:center;">
<table style="border:none; background:transparent; text-align:center;">
<tbody>
    <tr>
        <th>Name</th>
        <th>Description</th>
        <th>Complexity</th>
    </tr>
    <tr>
        <td> <b>Insert</b> </td>
        <td> Add an element to the list </td>
        <td> Time and Space: $O(1)$ </td>
    </tr>
    <tr>
        <td> <b>Remove</b> </td>
        <td> Remove an element from the list </td>
        <td> Time and Space: $O(1)$ </td>
    </tr>
    <tr>
        <td> <b>Search</b> </td>
        <td> Search an element in the list </td>
        <td> Time and Space: $O(N)$ </td>
    </tr>
</tbody>
</table>
</div>


```python
class Node:
    def __init__(self, value):
        self.value = value
        self.next = None

class LinkedList:
    # Initializing a stack.
    # Use a dummy node, which is
    # easier for handling edge cases.
    def __init__(self):
        self.head = None
        self.tail = None

    # Check if the stack is empty
    def isEmpty(self):
        return self.head is None

    # Insert at the end
    def insert(self, value):
        node = Node(value)
        if self.isEmpty():
            self.head = node
            self.tail = node
        else:
            self.tail.next = node
            self.tail = self.tail.next

    # remove from the beginning.
    def remove(self):
        if self.isEmpty():
            raise Exception("Remove from an empty list")

        node = self.head
        if self.head == self.tail:
            self.tail = self.tail.next
        self.head = self.head.next
        return node

    # sarch the node with a given value
    def search(self, value):
        node = self.head
        while node is not None:
            if node.value == value:
                break
            node = node.next
        return node
```

#### Floyd’s Cycle Finding Algorithm

One of the most famous algorithm for LinkedList is the so called Floyd's finding algorithm.
This algorithm is used to find a loop in a linked list. It uses two pointers one moving twice as fast as the other one.
The faster one is called the faster pointer and the other one is called the slow pointer.
While traversing the linked list one of these things will occur:

  - the fast pointer may reach the end (NULL) this shows that there is no loop in the linked list.
  - the fast pointer again catches the slow pointer at some time therefore a loop exists in the linked list.

```python
def detectLoop(llist):
    slow_pointer = llist.head
    fast_pointer = llist.head

    while (slow_pointer != None
           and fast_pointer != None
           and fast_pointer.next != None):
        slow_pointer = slow_pointer.next
        fast_pointer = fast_pointer.next.next
        if (slow_pointer == fast_pointer):
            return 1

    return 0
```


### Stack

The Stack is a special kind of linked list that follows the **LIFO** principle. Intuitivly it's a deck of cards where the top card of the deck (the last added element added) is the first card to picked (the first element to remove next).

<div style="text-align:center;">
<table style="border:none; background:transparent; text-align:center;">
<tbody>
    <tr>
        <th>Name</th>
        <th>Description</th>
        <th>Complexity</th>
    </tr>
    <tr>
        <td> <b>Push</b> </td>
        <td> Add an element to the top of a stack </td>
        <td> Time and Space: $O(1)$ </td>
    </tr>
    <tr>
        <td> <b>Pop</b> </td>
        <td> Remove an element from the top of a stack </td>
        <td> Time and Space: $O(1)$ </td>
    </tr>
    <tr>
        <td> <b>Peek</b> </td>
        <td> Get the value of the top element without removing it </td>
        <td> Time and Space: $O(1)$ </td>
    </tr>
    <tr>
        <td> <b>IsEmpty</b> </td>
        <td> Check if the stack is empty </td>
        <td> Time and Space: $O(1)$ </td>
    </tr>
</tbody>
</table>
</div>

```python
class Node:
    def __init__(self, value):
        self.value = value
        self.next = None
        self.prev = None


class Stack:
    # Initializing a stack.
    # Use a dummy node, which is
    # easier for handling edge cases.
    def __init__(self):
        self.head = None
        self.size = 0

    # Get the current size of the stack
    def getSize(self):
        return self.size

    # Check if the stack is empty
    def isEmpty(self):
        return self.size == 0

    # Get the top item of the stack
    def peek(self):
        # Sanitary check to see if we
        # are peeking an empty stack.
        if self.isEmpty():
            raise Exception("Peeking from an empty stack")
        return self.head.value

    # Push a value into the stack.
    def push(self, value):
        node = Node(value)
        if self.head is None:
            self.head = node
        else:
            node.prev = self.head
            self.head.next = node
            self.head = node
        self.size += 1

    # Remove a value from the stack and return.
    def pop(self):
        if self.isEmpty():
            raise Exception("Popping from an empty stack")
        remove = self.head
        self.head = self.head.prev
        self.head.next = None
        self.size -= 1
        return remove.value
```

### Queue

Queues are an implementation of LinkedList that follows the **FIFO** principle. Similarly to ticket queue outside a cinema hall, where the first person entering the queue is the first person who gets the ticket.

<div style="text-align:center;">
<table style="border:none; background:transparent; text-align:center;">
<tbody>
    <tr>
        <th>Name</th>
        <th>Description</th>
        <th>Complexity</th>
    </tr>
    <tr>
        <td> <b>Enqueue</b> </td>
        <td> Add an element to the end of the queue </td>
        <td> Time and Space: $O(1)$ </td>
    </tr>
    <tr>
        <td> <b>Dequeue</b> </td>
        <td> Remove an element from the front of the queue </td>
        <td> Time and Space: $O(1)$ </td>
    </tr>
    <tr>
        <td> <b>Peek</b> </td>
        <td> Get the value of the front of the queue without removing it </td>
        <td> Time and Space: $O(1)$ </td>
    </tr>
    <tr>
        <td> <b>IsEmpty</b> </td>
        <td> Check if the queue is empty </td>
        <td> Time and Space: $O(1)$ </td>
    </tr>
</tbody>
</table>
</div>

## Binary Trees
A binary tree is a tree data structure in which each parent node can have at most two children.
Each node of a binary tree consists of three items:
  - value of the node;
  - the address to the left child;
  - the address to the right child.

<div style="text-align:center;">
<table style="border:none; background:transparent; text-align:center;">
<tbody>
    <tr>
        <th>Name</th>
        <th>Description</th>
        <th>Complexity</th>
    </tr>
    <tr>
        <td> <b>Construct</b> </td>
        <td> Construct a binary tree </td>
        <td> Time and Space: $O(N)$ </td>
    </tr>
    <tr>
        <td> <b>Travers</b> </td>
        <td> Traverse a binary tree </td>
        <td> Time $O(N)$ and Space: $O(height)$ </td>
    </tr>
</tbody>
</table>
</div>

Binary trees are generaly represetned by linked nodes structures.

```python
class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

    def pre_order_traverse(self):
        print(self.value)
        if self.left:
            self.left.pre_order_traverse()
        if self.right:
            self.right.pre_order_traverse()

    def in_order_traverse(self):
        if self.left:
            self.left.in_order_traverse()

        print(self.value)

        if self.right:
            self.right.in_order_traverse()

    def post_order_traverse(self):
        if self.left:
            self.left.post_order_traverse()

        if self.right:
            self.right.post_order_traverse()

        print(self.value)

def build(array):
    root = None
    n = len(array)

    def add_child(idx):
        if idx < n:
            node = Node(array[idx])

            node.left = add_child(idx*2 + 1)

            node.right = add_child(idx*2 + 2)

        return node

    root = add_child(0)
    return root
```
However, binary trees can be also represented using arrays in which:
  - root node is stored at index 0;
  - left child is stored at index $$(i \cdot 2) + 1$$ where, i is the index of the parent;
  - right child is stored at index $$(i \cdot 2) + 1$$, i is the index of the parent.

```python
class Tree:
    def __init__(self, array):
        self.array = array

    def left(self, parent_idx):
        if (parent_idx * 2) + 1 < len(self.array):
            return self.array[(parent_idx * 2) + 1]

    def right(self, parent_idx):
        if (parent_idx * 2) + 2 < len(self.array):
            return self.array[(parent_idx * 2) + 2]

    def set_left(self, val, parent_idx):
        self.array[(parent_idx * 2) + 1] = val

    def set_right(self, val, parent_idx):
        self.array[(parent_idx * 2) + 2] = val

    def in_order(self, parent_idx=0):
        if self.left(parent_idx):
            self.in_order((parent_idx * 2) + 1)

        print(self.array[parent_idx])

        if self.right(parent_idx):
            self.in_order((parent_idx * 2) + 2)
```


### Binary Search Tree
Binary search tree is a data structure that quickly allows us to maintain a sorted list of numbers and search trought it.

The properties that separate a binary search tree from a regular binary tree are:
  - all nodes of left subtree are less than the root node;
  - all nodes of right subtree are more than the root node;
  - both subtrees of each node are also BSTs i.e. they have the above two properties.

Searching is extreamly efficent as can be done in $O(\log N)$ time and constant space.
Intuitively, searching is so efficent as we can analys only one of the two subtrees based on the relation between the current node's value and the looked for value.
Thus, at every step we half the searching space.

<div style="text-align:center;">
<table style="border:none; background:transparent; text-align:center;">
<tbody>
    <tr>
        <th>Name</th>
        <th>Description</th>
        <th>Complexity</th>
    </tr>
    <tr>
        <td> <b>Search</b> </td>
        <td> Search for an element in the tree </td>
        <td> Time $O(\log N)$ and Space: $O(N)$ </td>
    </tr>
    <tr>
        <td> <b>Insertion</b> </td>
        <td> Insert a node to the tree </td>
        <td> Time $O(\log N)$ and Space: $O(N)$ </td>
    </tr>
    <tr>
        <td> <b>Deletion</b> </td>
        <td> Remove an element from the tree </td>
        <td> Time $O(\log N)$ and Space: $O(N)$ </td>
    </tr>
</tbody>
</table>
</div>


```python
class Node:
    def __init__(self, value):
        self.value = value
        self.left, self.right = None, None

    def search(self, value):

        if self.value == value:
            return True
        elif value < self.value and self.value.left is not None:
            return self.value.left.search(value)
        elif value > self.value and self.value.right is not None:
            return self.value.right.search(value)
        else:
            return False
    
    @staticmethod
    def insert(node, value):
        if node is None:
            return Node(value)

        if value < node.value:
            node.left = Node.insert(node.left, value)
        else:
            node.right = Node.insert(node.right, value)

        return node

    @staticmethod
    def remove(node, value):
        if value < node.value:
            node.left = Node.remove(node.left, value)
        elif value > node.value:
            node.right = Node.remove(node.right, value)

        else:
            # case 1: it is a leaf node
            if node.left is None and node.right is None:
                return None

            # case 2: there is only 1 child
            elif node.left is not None and node.right is None:
                node.value = node.left.value
                node.left = None
            elif node.left is None and node.right is not None:
                node.value = node.right.value
                node.right = None

            # case 3: take as new node, the right child or the left node of the right child if it exist
            else:
                # get new min value from right
                temp = node.right
                prev = node
                while temp.left is not None:
                    prev, temp = temp, temp.left

                node.value = temp.value

                if prev != node:
                    prev.left = temp.right
                else:
                    node.right = temp.right
            return node
        return node
```