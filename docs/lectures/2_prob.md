# Probability refresher

Another set of fundamental mathematical tools required to develop various machine learning algorithms 
(especially towards the end of the course when we will focus on generative modelling)

In order to develop various machine learning algorithms (especially towards the end of the 
course when we will focus on generative modelling) we need to familarize with some basic concepts of:
mathematical tools from:

- **Probability**: mathematical framework to handle uncertain statements;
- **Information Theory**: scientific field focused on the quantification of amount of uncertainty in a probability distribution.

## Probability

**Random Variable**: a variable whose value is unknown, all we know is that it can take on different 
values with a given probability. It is generally defined by an uppercase letter $X$, whilst the values 
it can take are in lowercase letter $x$.

**Probability distribution**: description of how likely a variable $x$ is, $P(x)$ (or $p(x)$). 
Depending on the type of variable we have:

- *Discrete distributions*: $P(X)$ called Probability Mass Function (PMF) and $X$ can take on a discrete number of states N.
A classical example is represented by a coin where N=2 and $X={0,1}$. For a fair coin, $P(X=0)=0.5$ and $P(X=1)=0.5$.

- *Continuous distributions*: $p(X)$ called Probability Density Function (PDF) and $X$ can take on any value from a continuous space 
  (e.g., $\mathbb{R}$). A classical example is represented by the gaussian distribution where $x \in (-\infty, \infty)$.
  
A probability distribution must satisfy the following conditions:

- each of the possible states must have probability bounded between 0 (no occurrance) and 1 (certainty of occurcence):
  $\forall x \in X, \; 0 \leq P(x) \leq 1$ (or $p(x) \geq 0$, where the upper bound is removed because of the 
  fact that the integration step $\delta x$ in the second condition can be smaller than 1: $p(X=x) \delta x <=1$);

- the sum of the probabilities of all possible states must equal to 1: $\sum_x P(X=x)=1$ (or $\int p(X=x)dx=1$).

**Joint and Marginal Probabilities**: assuming we have a probability distribution acting over a set of variables (e.g., $X$ and $Y$) 
we can define

- *Joint distribution*: $P(X=x, Y=y)$ (or $p(X=x, Y=y)$);

- *Marginal distribution*: $P(X=x) = \sum_{y \in Y} P(X=x, Y=y)$ (or $p(X=x) = \int P(X=x, Y=y) dy$), 
  which is the probability spanning one or a subset of the original variables;

**Conditional Probability**: provides us with the probability of an event given the knowledge 
that another event has already occurred

$$
P(Y=y | X=x) = \frac{P(X=x, Y=y)}{P(X=x)}
$$

This formula can be used recursively to define the joint probability of a number N of variables as product of conditional
probabilities (so-called *Chain Rule of Probability*)

$$
P(x_1, x_2, ..., x_N) = P(x_1) \prod_{i=2}^N P(x_i | x_1, x_2, x_{i-1})
$$

**Independence and Conditional Independence**: Two variables X and Y are said to be indipendent if

$$
P(X=x, Y=y) = P(X=x) P(Y=y)
$$

If both variables are conditioned on a third variable Z (i.e., P(X=x, Y=y | Z=z)), they are said to be conditionally
independent if

$$
P(X=x, Y=y | Z=z) = P(X=x | Z=z) P(Y=y| Z=z)
$$

**Mean (or Expectation)**: Given a function $f(x)$ where $x$ is a stochastic variable with probability $P(x)$, its average
or mean value is defined as follows for the discrete case:

$$
\mu = E_{x \sim P} [f(x)] = \sum_x P(x) f(x)
$$

and for the continuos case

$$
\mu = E_{x \sim p} [f(x)] = \int p(x) f(x) dx
$$

In most Machine Learning applications we do not have knowledge of the full distribution to evaluate the mean, rather we 
have access to N equi-probable samples that we assume are drawn from the underlying distribution. We can approximate the mean
via the *Sample Mean*:

$$
\mu \approx \sum_i \frac{1}{N} f(x_i)
$$

**Variance (and Covariance)**: 

## Information theory
