# Linear and Logistic Regression

In the previous lecture we have learned how to optimize a generic loss function $J_\theta$ 
by modifying its free parameters $\theta$. Whilst this is a very generic framework that can be used for 
various applications in different scientific field, from now on we will learn how to take advtange of
similar algorithms in the context of Machine Learning.

## Linear regression
In preparation to our lecture on Neural Networks, we consider here what is generally referred to as the simplest
machine learning model for regression, *linear regression*. Its simplicity lies in the fact that we will only consider 
a linear relationship between our inputs and targets:

![LIN REG](figs/reg_model.png)

where $\textbf{x}$ is a training sample with $N_f$ features, $\textbf{w}$ is a vector of $N_f$ weights and $b=w_0$ is the
so-called bias term. The set of trainable parameters is therefore the combination of the weights and bias 
$\boldsymbol\theta=[\textbf{w}, b] \in \mathbb{R}^{N_f+1}$. The prediction $\hat{y}$ is simply obtained by linearly 
combining the different features of the input vector and adding the bias.

Assuming availability of $N_s$ training samples, the **model** can be compactly written as:

$$
\hat{\textbf{y}}_{train} = \textbf{X}_{train}^T \boldsymbol\theta
$$

where $\textbf{X}_{train} \in \mathbb{R}^{N_f+1 \times N_s}$ is the training matrix and $\hat{\textbf{y}}_{train} \in \mathbb{R}^{1 \times N_s}$ 
is the predicted target vector. Note that each column of the training matrix contains the corresponding training sample followed by a 1 
(i.e., $\textbf{X}_{:,j}=[\textbf{x}^{(j)}, 1]$).

Next, we need to define a metric (i.e., cost function) which we can use to optimize for the free parameters $\boldsymbol\theta$.
For regression problems, a common metric of goodness is the L2 norm or MSE (Mean Square Error):

$$
J_\theta = MSE(\textbf{y}_{train}, \hat{\textbf{y}}_{train}) = || \textbf{y}_{train} - \hat{\textbf{y}}_{train}||_2^2 = 
\frac{1}{N_s} \sum_i^{N_s} (y_{train}^{(i)}-\hat{y}_{train}^{(i)})^2
$$

Based on our previous lecture on optimization, we need to find the best set of coefficients $\theta$ that minimize the MSE:

$$
\hat{\theta} = min_\theta  J_\theta \rightarrow \theta_{i+1} = \theta_i - \alpha \nabla J_\theta
$$

However, since this is a linear inverse problem we can write the analytical solution of the minimization problem as:

$$
\hat{\theta} = (\textbf{X}_{train}^T \textbf{X}_{train})^{-1} \textbf{X}_{train}^T \textbf{y}_{train}
$$

which can be obtained by inverting a $N_f+1 \times N_f+1$ matrix. 

An important observation, which lies at the core of most Machine Learning algorithms, is that once the model is trained 
on the $N_s$ available input-target pairs, the estimated \hat{\theta} coefficients can be used to make *inference* on any new unseen data:

$$
y_{test} = \textbf{x}^T_{test} \hat{\theta}
$$

## Logistic regression

HERE!!!!

Log regr: derivative derivation and implementation, look here https://towardsdatascience.com/logistic-regression-from-scratch-69db4f587e17

Explain that analytic expression for gradient isnt always possible, nor the most efficient... we will show later a more general approach to it.

Mention that in next class we will understand why we choose such cost functions