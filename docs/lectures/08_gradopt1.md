# More on gradient-based optimization

Whilst stochastic gradient descent is a easy to understand, and simple to implement algorithm (as discussed in this
[lecture](lectures/03_gradopt.md)), it presents a number of shortcomings that prevent learning to be as fast and effective
as we would like it to be. In this lecture, we will discuss some of the limitations of SGD and look at alternative optimization
algorithms that have been developed in the last decade and are nowadays preferred to SGD in the process of training NNs.

## Ill-conditioning

The shape, and more specifically the curvature, of the functional that we wish to minimize affects
our ability to quickly and efficiently converge to one of its minima (ideally the global, likely one of the local). For nonlinear optimization problems,
like those encountered in deep learning, this is mathematically represented its *Hessian* matrix 
($\mathbf{H}=\frac{\partial^2 f}{\partial \boldsymbol \theta^2}$). An Hessian matrix with large conditioning number (i.e.,
ratio of the largest and smallest eigenvalues) tends to affect convergence speed of first-order (gradient-based) methods.

In classical optimization theory, second order methods such as the Gauss-Netwon method are commonly employed to counteract
this problem. However, as already mentioned in one of our previous lectures, such methods are not yet suitable for deep learning
in that no mathematical foundations have been developed in conjunction with approximate gradients (i.e., mini-batch learning
strategy). 

Another factor that is worth knowing about is related to the norm of the gradient $\mathbf{g}^T\mathbf{g}$ through iterations.
In theory this norm should shrink through iterations to guarantee convergence. Nevertheless, successful training may still be 
obtained even if the norm does not shrink as long as the learning rate is kept small. Let's write the second-order Taylor
expansion of the functional around the current parameter estimate $\boldsymbol \theta_0$:

$$
J(\boldsymbol \theta) \approx J(\boldsymbol \theta_0) + (\boldsymbol \theta - \boldsymbol \theta_0)^T \mathbf{g} + 
\frac{1}{2} (\boldsymbol \theta - \boldsymbol \theta_0)^T \mathbf{H} (\boldsymbol \theta - \boldsymbol \theta_0)
$$

and evaluate it at the next gradient step $\boldsymbol \theta = \boldsymbol \theta_0 - \alpha \mathbf{g}$:

$$
J(\boldsymbol \theta_0 - \alpha \mathbf{g}) \approx J(\boldsymbol \theta_0) - \mathbf{g}^T \mathbf{g} + 
\frac{1}{2} \alpha^2 \mathbf{g}^T \mathbf{H} \mathbf{g}
$$

We can interpret this expression as follows: a gradient step of $- \alpha \mathbf{g}$ adds the following contribution
to the cost function, $-\mathbf{g}^T \mathbf{g} + 
\frac{1}{2} \alpha^2 \mathbf{g}^T \mathbf{H} \mathbf{g}$. When this contribution is positive (i.e., 
$\frac{1}{2} \alpha^2 \mathbf{g}^T \mathbf{H} \mathbf{g} > \mathbf{g}^T \mathbf{g}$), the cost function grows instead of
being reduced. Under the assumption that $\mathbf{H}$ is known, we could easily choose a step-size $\alpha$ that prevents this from happening.
However, when the Hessian cannot be estimated, a conservative selection of the step-size is the only remedy to prevent the cost function
from growing. A downside of such an approach is that the smaller the learning rate the slower the training process.


## Local minima

Whilst the focus of the previous section has been in the neighbour of $\boldsymbol \theta_0$ where the functional 
$J_{\boldsymbol \theta}$ can be approximated by a convex function, the landscape of NN functionals is generally non-convex
and populated with a multitude of local minima. The problem of converging to the global minimum without getting stuck 
in one of the local minima is a well-known problem for any non-convex optimization. An example in geophysics is represented
by waveform inversion and a large body of work has been carried out by the geophysical research community to identify
objective functions that are more well-behaved (i.e., show a large basin of attraction around the global minimum).

Nevertheless, getting stuck into local minima is much less of a problem when training neural networks. 
This can be justified by the fact that multiple models may perform equally well on both the training and testing data. 
To be more precise this relates to the concept of *model identifiability*, where a model is defined identifiable if there exist a 
single set of parameters ($\boldsymbol \theta_{gm}$) that lead to optimal model performance. On the other hand, when multiple models $\{ \boldsymbol \theta_{gm}, 
\boldsymbol \theta_{lm,1}, ..., \boldsymbol \theta_{lm1,N}$ perform similarly those models are said to be non-identifiable. Moroever, even when a 
single model performs best, a distinction must be made between training and testing performance. As far as training performance is concerned,
this model must be that of the global minimum of the functional $\boldsymbol \theta_{gm}$. Nevertheless, the model that performs best on the testing
data may be the one obtained from any of the local minima $\boldsymbol \theta_{lm,i}$ as such a model be have better generalization capabilities
than the one from the global minimum. Moreover, recent research in the field of deep learning has revealed that such multi-dimensional landscapes may actually
have much fewer local minima than we tend to believe, and the main hinder to slow training is actually represented by saddle points as discussed in the next section.

## Saddle points and other flat regions
