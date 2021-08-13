# Gradient-based optimization
After reviewing some of the basic concepts of linear algebra that we will be using during this course, we are now in a position
to start our journey in the field of *learning algorithms*. Any learning algorithm, no matter its level of complexity, is 
composed of 4 key elements:

**Dataset**: a collection of many examples (sometimes referred to as samples of data points) that represents the experience
we wish our machine learning algorithm to learn from. More speficically, the dataset is defined as:
$$
\mathbf{x} = [x_1, x_2, ..., x_{N_f}]^T \quad \mathbf{X} = [\mathbf{x}^{(1)}, \mathbf{x}^{(2)}, ..., \mathbf{x}^{(N_s)}] 
$$
and
$$
\mathbf{y} = [y_1, y_2, ..., y_{N_t}]^T \quad \mathbf{Y} = [\mathbf{y}^{(1)}, \mathbf{y}^{(2)}, ..., \mathbf{y}^{(N_s)}] 
$$
where $N_f$ and $N_t$ are the number of features and targets for each sample in the dataset, respectively, and 
$N_s$ is the number of samples.

**Model**: a mathematical relation between the input (or features) and output (or target) 
or our dataset. It is generally parametrized as function $f$ of a number of free parameters $\theta$ which we want the 
learning algorithm to estimate given a task and a measure of performance, and we write it as 
$$
\mathbf{y} = f_\theta(\mathbf{x})
$$

**Loss (and cost) function**: quantitative measure of the performance of the learning algorithm, which we wish to minimize 
(or maximize) in order to make accurate predictions on unseen data. It is written as
$$
J = \sum_{i=1}^{N_s}  \mathscr{L} (\mathbf{y}^{(i)}, f(\mathbf{x}^{(i)}))
$$

where $\mathscr{L}$ is the loss function for each input-output pair and $J$ is the overall cost function.

**Optimization algorithm**: mathematical method that modifies the aims to drive down (up) the cost function by modifying
its free-parameters $\theta$:
$$
\hat{\theta} = \underset{\theta} {\mathrm{argmin}} J
$$

Optimization algorithms are generally divided into two main families: gradient-based 
(or local) and gradient-free (or global). Gradient-based optimization is by far the most popular way to train NNs and 
will be discussed in more details below.

## Gradient- and steepest-descent algorithms
The simplest of gradient-based methods is the so-called **Gradient-descent** algorithm. As the name implies, 
this algorithm uses local gradient information of the functional to minimize/maximize to move towards its global 
mimimum/maximum as depicted in the figure below.

![GRADIENT OPTIMIZATION](figs/opt_gradient.png)

More formally, given a functional $f_\theta$ and its gradient $\nabla f = \frac{\delta f}{\delta \theta}$, 
the (minimization) algorithm can be written as:

Initialization: choose $\theta \in \mathbb{R}$

For $i=0,...N-1$;

1. Compute update direction $d_i = -\nabla f |_{\theta_i}$
2. Estimate step-lenght $\alpha_i$
3. Update $\theta_{i+1} = \theta_{i} + \alpha_i d_i$

Note that the maximization version of this algorithm simply switches the sign in the update direction (first equation of the algorithm).
Moreover, the proposed algorithm can be easily extended to N-dimensional model vectors $\theta=[\theta_1, \theta_2, ..., \theta_N]$ by
defining the following gradient vector 
$\nabla f=[\delta f / \delta\theta_1, \delta f / \delta\theta_2, ..., \delta f/ \delta\theta_N]$.

### Step lenght selection
The choice of the step-lenght has tremendous impact on the performance of the algorithm and its ability to converge 
fast (i.e., in a small number of iterations) to the optimal solution.

The most used selection rules are:

- Constant: the step size is fixed to a constant value $\alpha_i=\hat{\alpha}$. This is the most common situation that we
  will encounter when training neural networks. In practice, some adaptive schemes based on the evolution of the train
  (or validation) norm are generally adopted, but we will still refer to this case as costant step size;
- Exact linesearch: at each iteration, $\alpha_i$ is chosen such that it minimizes $f(\theta_{i} + \alpha_i d_i)$. This
  is the most commonly used approach when dealing with linear systems of equations.
- Backtracking  "Armijo" linesearch: at each iteration, given a parameter $\mu \in (0,1)$, start with $\alpha_i=1$ 
  and reduce it by a factor of 2 until the following condition is satisfied: $f(\theta_i) - f(\theta_{i} + \alpha_i d_i) \ge  -\mu \alpha_i \nabla f^T d_i$
  
## Second-order optimization
Up until now we have discussed first-order optimization techniques that rely on the ability to evaluate the function $f$ and 
its gradient $\nabla f$. Second-order optimization method go one step beyond in that they use information from both the 
local slope and curvature of the function f. 

When a function has small curvature, the function and its tangent line are very similar: 
the gradient alone is therefore able to provide a good local approximation of the function (i.e., $f(\theta+\delta \theta)\approx f(\theta) + \nabla f \delta \theta$).
On the other hand, if the curvature of the function of large, the function and its tangent line start to differ very quickly away from
the linearization point. The gradient alone is not able anymore to provide a good local approximation of the function 
(i.e., $f(\theta+\delta \theta)\approx f(\theta) + \nabla f \delta \theta + \nabla^2 f \delta \theta^2$).

Let's start again from the one-dimensional case and the well-known **Newton's method**. This method is generally employed to find the zeros of a function:
$x: f(x)=0$ and can be written as:

$$
x_{i+1} = x_i - \frac{f(x)|_{x_i}}{f'(x)|_{x_i}} 
$$

which can be easily derived from the Taylor expansion of $f(x)$ around $x_{i+1}$.

If we remember that finding the minimum (or maximum) of a function is equivalent to find the zeros of its first derivative 
($x: min_x f(x) \leftrightarrow x: f'(x)=0$), the Netwon's method can
be written as:

$$
x_{i+1} = x_i - \frac{f'(x)|_{x_i}}{f''(x)|_{x_i}} 
$$

This is the so-called **Gauss-Netwon method**.

In order to be able to discuss second-order optimization algorithms for the multi-dimensional case, let's first introduce the notion of *Jacobian*:

$$\mathbf{y} = f(\mathbf{x}) \rightarrow  \mathbf{J}  = \begin{bmatrix} 
                \frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & ... & \frac{\partial f_1}{\partial x_M} \\
                ...     & ...  & ...   & ... \\
                \frac{\partial f_N}{\partial x_1} & \frac{\partial f_N}{\partial x_2} & ... & \frac{\partial f_N}{\partial x_M} \\
  \end{bmatrix} \in \mathbb{R}^{[N \times M]}
$$

Through the notion of Jacobian, we can define the **Hessian** as the Jacobian of the gradient vector

$$\mathbf{H} = \nabla (\nabla f) = \begin{bmatrix} 
                \frac{\partial f^2}{\partial x_1^2} & \frac{\partial f^2}{\partial x_1 \partial x_2 } & ... & \frac{\partial f^2}{\partial x_1\partial x_M} \\
                ...     & ...  & ...   & ... \\
                \frac{\partial f^2}{\partial x_M \partial x_1} & \frac{\partial f^2}{\partial x_M \partial x_2} & ... & \frac{\partial f^2}{\partial x_M^2} \\
  \end{bmatrix} \in \mathbb{R}^{[M \times M]}
$$

where we note that when $f$ is continuous, $\partial / \partial x_i \partial x_j = \partial / \partial x_j \partial x_i$, and $\mathbf{H}$
is symmetric.

The Gauss-Newton method can be written as:

$$
\mathbf{x}_{i+1} = \mathbf{x}_i - \mathbf{H}^{-1}\nabla f
$$

Approximated version of the Gauss-Netwon method have been developed over the years, mostly based on the idea that inverting $\mathbf{H}$ is
sometimes a prohibitive task. Such methods, generally referred to as Quasi-Netwon methods attempt to approximate the Hessian (or its inverse)
using the gradient at the current iteration and that of a number of previous iterations. [BFGS](https://en.wikipedia.org/wiki/Broyden–Fletcher–Goldfarb–Shanno_algorithm)
or its limited memory version [L-BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS) are examples of such a kind. Due to their computational cost
(as well as the lack of solid theories for their use in conjunction with approximate gradients), these methods are not yet commonly used by the
machine learning community to optimize the parameters of NNs in deep learning.

## Stochastic-gradient descent

!!HERE!!


Finally, I encourage everyone to read the following [blog post](https://ruder.io/optimizing-gradient-descent/) for a more
detailed overview of the optimization algorithms discussed here. Note that we will also look more into some of the recent 
optimization algorithms that overcome some of the limitations of standard SGD in [this lecture](lectures/10_gradopt1.md).
