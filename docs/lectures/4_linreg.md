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
$\boldsymbol\theta=[\textbf{w}, b] \in \mathbb{R}^{N_f+1}$. Similarly the combination of the training sample and a 1-scalar is defined as 
$\tilde{\textbf{x}}=[\textbf{x}, 1] \in \mathbb{R}^{N_f+1}$
The prediction $\hat{y}$ is simply obtained by linearly 
combining the different features of the input vector and adding the bias.

Despite its simplicity, linear regression (and more commonly multi-variate linear regression) has been succesfully used in 
a variety of geoscientific tasks, examples of such a kind are:

- rock-physics models, where a number of petrophysical parameters (e.g., porosity, shale content, depth) can be linearly regressed 
  in order to predict an elastic parameter of interest (e.g., dry bulk modulus);
  
- time-to-depth conversion, where a velocity (or depth) prediction is generally made as a linear combination of two-way traveltime
  and other parameters such as seismic amplitudes and various derived attributes;
  
- filling gaps in petrophysical well logs, where various petrophysical measurements (e.g., GR, NEU, DEN) are regressed to 
  estimate another quantity of interest (e.g., S-wave velocity of DTS) that is not directly available within a certain depth interval.

Assuming availability of $N_s$ training samples, the input training matrix and output training vector of a linear regression 
model is written as:

$$
\mathbf{X}_{train} = [\tilde{\mathbf{x}}^{(1)}, \tilde{\mathbf{x}}^{(2)}, ..., \tilde{\mathbf{x}}^{(N_s)}] \in \mathbb{R}^{N_f+1 \times N_s}, \quad
\mathbf{y}_{train} = [y^{(1)}, y^{(2)}, y^{(N_s)}] \in \mathbb{R}^{N_s \times 1}
$$

Finally, the **model** can be compactly written as:

$$
\hat{\textbf{y}}_{train} = \textbf{X}_{train}^T \boldsymbol\theta
$$

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

which can be obtained by inverting a $N_s \times N_s$ matrix. 

An important observation, which lies at the core of most Machine Learning algorithms, is that once the model is trained 
on the $N_s$ available input-target pairs, the estimated $\hat{\theta}$ coefficients can be used to make *inference* on any new unseen data:

$$
y_{test} = \tilde{\textbf{x}}^T_{test} \hat{\theta}
$$

To conclude, once a linear regression model has been trained, a variety of measures exist to assess the goodness of the model. Whilst the same 
metric used for training, the mean-square error, can be used to assess the model performance, other metrics are represented by the Pearson coefficient 
($R^2$) and the mean-absolute erorr (MAE).

## Logistic regression

Simply put, logistic regression is an extension of linear regression to the problem of binary classification. Whilst the model used
by logistic regression is the same linear model described above, this will be coupled with a nonlinear 'activation' function that enforces the
outcome of the entire model to be bounded between 0 and 1 (i.e., a probability). In other words, whilst the input training matrix is the same
as that of linear regression, the output training vector becomes:

$$
y_{train} = \{0, 1\}
$$

A variety of applications of such a simple model can be found in geoscience, one common example is represent by net pay prediction 
from petrophysical logs.

Given a single pair of training samples $\textbf{x}, y$, a mathematical **model** for logistic regression can be compactly written as:

$$
\hat{y} = f_\theta(\textbf{x}) = P(y=1 | \textbf{x}) \in (0,1)
$$

or in other words, the input vector $\textbf{x}$ is fed through a nonlinear model $f_\theta$ whose output is a scalar number between 0 and 1 that 
represents the probability of the target output to be 1.

Considering now a set of $N_s$ training pairs, the model can be explicitely written as:

$$
\hat{\textbf{y}}_{train} = f_\theta(\textbf{X}_{train}) = \sigma(\textbf{X}_{train}^T \boldsymbol\theta)
$$

where $\sigma$ is a sigmoid function as shown in figure below:

![SIGMOID](figs/sigmoid.png)

Once again, let's define a cost function that we can use to optimize the model parameters. For binary classification, a common metric of goodness
is represented by the so-called *binary cross-entropy*:

$$
\mathscr{L}(y_{train}^{(i)}, \hat{y}_{train}^{(i)}) = -(y_{train}^{(i)} log(\hat{y}_{train}^{(i)}) + 
(1-y_{train}^{(i)})) log(1- \hat{y}_{train}^{(i)}))
$$

and

$$
J_\theta = \frac{1}{N_s} \sum_i^{N_s} \mathscr{L}(y_{train}^{(i)}, \hat{y}_{train}^{(i)})
$$

Let's gain some intuition onto why this is a good cost function. More specifically, we consider with a drawing the two cases separately.
First the case of positive target, $y_{train}^{(i)}=1$ 

![LOGISTIC POSITIVE](figs/logistic_pos.png)

and then the case of negative target, $y_{train}^{(i)}=0$:

![LOGISTIC NEGATIVE](figs/logistic_neg.png)

Our drawings cleary show the validity of such a cost function in both cases. The further away is the prediction from the true label the higher the
resulting cost function. Similar to the case of linear regression, we can now update the model parameters by minimizing the cost function:

$$
\hat{\theta} = min_\theta  J_\theta \rightarrow \theta_{i+1} = \theta_i - \alpha \nabla J_\theta
$$

However a major difference arises here. Whilst it is easy to compute the derivative of the MSE with respect to the model parameters $\theta$, and 
even more since the model is linear an analytical solution can be found (as shown above), this is not the case of the cost function of the logistic
regression model.

The good news here is that there exist a systematic approach to computing the derivative of a composite function (i.e., $f(x)=f_N(...f_2(f_1(x)))$), which
simply relies on the well-known *chain rule* of functional analysis. This method is referred to in the mathematical community as Automatic Differentiation (AD),
and more likely so as Back-propagation in the ML community. As this lies as the foundation of the training process for neural networks, we will get into details 
later in the text. At this point suffices to say that if we have a composite function like the one above, its derivative with respect to $x$ can be written as:

$$
\frac{\partial f}{\partial x} = \frac{\partial f_N}{\partial f_{N-1}} ... \frac{\partial f_2}{\partial f_1} \frac{\partial f_1}{\partial x}
$$

where the derivative is simply the product of a number of derivatives over the chain of operations of the composite function.
Note that in practice it is more common to compute this chain rule in reverse order, from left to right in the equation above.

Whilst we generally rely on the built-in functionalities of deep learning libraries such as Tensorflow or PyTorch to compute such derivaties, we will
perform here a full derivation for the simple case of logistic regression. In order to do so, we introduce a very useful mathamatical tool that we use to 
keep track of a chain of operations and later know how to evaluate the associated gradient. This tool is usually known as *computational graph*. More specifically,
instead of writing the entire logistic regression model compactely in a single equation, we divide it here into its atomic components:

$$
z = \textbf{x}^T \boldsymbol\theta, \quad
a = \sigma(z), \quad \mathscr{L} = -(y log(a) + (1-y)log(1-a))
$$

such that the derivative of the loss function with respect to the model parameters becomes:

$$
\frac{\partial \mathscr{L} }{\partial \boldsymbol\theta} = \frac{\partial \mathscr{L} }{\partial a}
\frac{\partial a }{\partial z} \frac{\partial z}{\partial \boldsymbol\theta}
$$

The forward and backward passes (as described in software frameworks like PyTorch) can be visually displayed as follows:

![BACKPROP](figs/backprop.png)

Let's start from $\partial \mathscr{L} \ \partial a$:

$$
\frac{\partial \mathscr{L}}{\partial a} = -\frac{y}{a} + \frac{1-y}{1-a} = \frac{-y(1-a) + (1-y)a}{a (1-a)}
$$

and $\partial a \ \partial \sigma$:

$$
\frac{\partial a}{\partial z} = a(1-a)
$$

which we can combine together to obtain a simplified formula for the derivative of the loss function of the output of the weighted summation ($z$)

$$
\frac{\partial \mathscr{L}}{\partial z} = \frac{\partial \mathscr{L}}{\partial a} \frac{\partial a}{\partial \sigma} = 
-y(1-a) + (1-y)a = a - y = dz
$$

Finally we differentiate between the weights and the bias to obtain:

$$
\frac{\partial z}{\partial w_i} = x_i, \quad \frac{\partial z}{\partial b} = 1
$$

such that:

$$
\frac{\partial \mathscr{L}}{\partial w_i} = dz \cdot x_i = dw_i, \quad \frac{\partial \mathscr{L}}{\partial b} = dz = db
$$

Having found the gradients, we can now update the parameters as discussed above:

$$
w_i \leftarrow w_i - \alpha \frac{\partial \mathscr{L}}{\partial w_i} = w_i - \alpha dw_i, \quad
b \leftarrow b - \alpha \frac{\partial \mathscr{L}}{\partial b} = b - \alpha db
$$

which can be easily modified in the case of multiple training samples:

$$
w_i \leftarrow  w_i - \alpha \sum_{j=1}^{N_s} dw_i^{(j)}, \quad
b \leftarrow b - \alpha \sum_{j=1}^{N_s} db^{(j)}
$$

We can now summarize a single step of training for $N_s$ training samples for the logistic regression model:

$\textbf{z}=\textbf{X}_{train}^T \boldsymbol \theta$

$\textbf{a} = \sigma(\textbf{z})$

$\textbf{dz} = \textbf{a} - \textbf{y}$

$\textbf{dw} = \frac{1}{N_s} \textbf{X}_{train} \textbf{dz}$

$db = \frac{1}{N_s} \textbf{1}^T \textbf{dz}$

$\textbf{w} \leftarrow \textbf{w} - \alpha \textbf{dw}$

$b \leftarrow b - \alpha db$

To conclude, let's turn our attention into some of the *evaluation metrics* that are commonly used to assess the performance of a 
classification model (or classifier). Note that these metrics can be used for the logistic regression model discussed here as well as for
other more advanced models discussed later in the course. 

In general for binary classification we have two possible outcomes (positive/negative or true/false) for both the true labels $y$ and the 
predicted labels $\hat{y}$. We can therefore define 4 complementary metrics as shown in the figure below:


