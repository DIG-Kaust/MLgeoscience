# Linear Algebra refresher

In this lecture we will go through some of the key concepts of linear algebra and inverse problem theory that are required
to develop the theories of the different machine learning algorithm presented in this course. 
This is not meant to be an exhaustive treatise and students are strongly advised to take the
[ErSE 213 - Inverse Problems](https://academicaffairs.kaust.edu.sa/Courses/Pages/DownloadSyllabus.aspx?Year=2017&Semester=030&Course=00008206&V=I)
prior to this course.

Three key mathematical objects arise in the study of linear algebra:

**Scalars**: $a \in \mathbb{R}$, a single number represented by a lower case italic letter;

**Vectors**: $\mathbf{x} = [x_1, x_2, ..., x_N]^T \in \mathbb{R}^N$, ordered collection of $N$ numbers
represented by a lower case bold letter; it is sometimes useful to extract a subset of elements by defining a set
$\mathbb{S}$ and add it to as a superscript, $\mathbf{x}_\mathbb{S}$. As an example, given $\mathbb{S} = {1, 3, N}$
we can define the vector $\mathbf{x}_\mathbb{S} = [x_1, x_3, ..., x_N]$ and its complementary vector 
$\mathbf{x}_{-\mathbb{S}} = [x_2, x_4, ..., x_{N-1}]$

**Matrices**: $\mathbf{X} \in \mathbb{R}^{[N \times M]}$, two dimensional collection of numbers represented by an upper case bold letter
where $N$ and $M$ are referred to as the height and width of the matrix. More specifically a matrix can be written as
  
$$\mathbf{X} = \begin{bmatrix} 
                x_{1,1} & x_{1,2} & x_{1,M} \\
                ...     & ...     & ... \\
                x_{N,1} & x_{N,2} & x_{N,M}
  \end{bmatrix}
$$

A matrix can be indexed by rows $\mathbf{X}_{i, :}$ (i-th row), by columns $\mathbf{X}_{:, j}$ (j-th column), and by 
element $\mathbf{X}_{i, j}$ (i-th row, j-th column). A number of useful operations that are commonly applied on vectors 
and matrices are now described: 

- Transpose: $\mathbf{Y} = \mathbf{X}^T$, where $Y_{i, j} = X_{j, i}$
- Matrix plus vector: $\mathbf{Y}_{[N \times M]} = \mathbf{X}_{[N \times M]} + \mathbf{z}_{[1 \times M]}$, where 
  $Y_{i, j} = X_{i, j} + z_{j}$ ($\mathbf{z}$ is added to each row of the matrix $\mathbf{X}$)
- Matrix-vector product: $\mathbf{y}_{[N \times 1]} = \mathbf{A}_{[N \times M]} \mathbf{x}_{[M \times 1]}$, where 
  $y_i = \sum_{j=1}^M A_{i, j} x_j$
- Matrix-vector product: $\mathbf{y}_{[N \times 1]} = \mathbf{A}_{[N \times M]} \mathbf{x}_{[M \times 1]}$, where 
  $y_i = \sum_{j=1}^M A_{i, j} x_j$
- Matrix-matrix product: $\mathbf{C}_{[N \times K]} = \mathbf{A}_{[N \times M]} \mathbf{B}_{[M \times K]}$, where 
  $C_{i,k} = \sum_{j=1}^M A_{i, j} B_{j, k}$
- Hadamart product (i.e., element-wise product): $\mathbf{C}_{[N \times M]} = \mathbf{A}_{[N \times M]} \odot \mathbf{B}_{[N \times M]}$, where 
  $C_{i,j} = A_{i, j} B_{i, j}$
- Dot product: $a = \mathbf{x}_{[N \times 1]}^T \mathbf{y}_{[N \times 1]} = \sum_{i=1}^N x_i y_i$
- Identity matrix: $\mathbf{I}_N = diag\{\mathbf{1}_N\}$. Based on its definition, we have that 
  $\mathbf{I}_N \mathbf{x} = \mathbf{x}$ and $\mathbf{I}_N \mathbf{X} = \mathbf{X}$
- Inverse matrix: given $\mathbf{y} = \mathbf{A} \mathbf{x}$, the inverse matrix of $\mathbf{A}$ is a matrix that
  satisfies the following equality $\mathbf{A}^{-1} \mathbf{A} = \mathbf{I}_N$. We can finally write
  $\mathbf{x} = \mathbf{A}^{-1} \mathbf{y}$
- Orthogonal vectors and matrices: given two vectors $\mathbf{x}$ and $\mathbf{y}$, they are said to be orthogonal if
  $\mathbf{y}^T \mathbf{x} = 0$. Given two matrices $\mathbf{X}$ and $\mathbf{Y}$, they are said to be orthogonal if
  $\mathbf{Y}^T \mathbf{X} = \mathbf{I}_N$. Orthogonal matrices are especially interesting because their inverse is very 
  cheap $\mathbf{X}^{-1} = \mathbf{X}^T$
  
Another important object that we will be using when defining cost functions for ML models is the so called **Norm**.