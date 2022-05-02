# Generative Modelling and Variational AutoEncoders

Up until now, our attention has been mostly focused on supervised learning tasks where we have access to a certain number
of training samples, in the form of input-target pairs, and we train a model (e.g., a NN) to learn the best possible mapping
between the two. These kind of models are also usually referred to as *discriminative models* as they learn from training samples
their underlying conditional probability distribution $p(\mathbf{y}|\mathbf{x})$.

In the last lecture, we have also seen how the general principles of supervised learning can be adapted to accomplish a
number of different tasks where input-target pairs are not available. Dimensionality reduction is one of such tasks, which are 
usually categorized under the umbrella of unsupervised learning.

Another very exciting area of statistics that has been recently heavily influenced by the deep learning revolution is the
so-called field of *Generative modelling*. Here, instead of having access to input-target pairs, we are able to only gather
a (large) number of samples $\mathbf{X} = \{ \mathbf{x}^{(1)}, \mathbf{x}^{(2)}, ..., \mathbf{x}^{(N_s)} \}$
that we believe come from a given hidden distribution. The task that we wish to accomplished is  therefore:

- Learn the underlying distribution $p(\mathbf{x})$, or 
- Learn to sample from the underlying distribution $\tilde{\mathbf{x}} \sim p(\mathbf{x})$

Obviously, the first task is more general and usually more ambitious. Once you know a distribution, sampling from it is rather an
easy task. In the next two lectures, we will however mostly focused on the second task and discuss two popular algorithms that
have shown impressive capabilities to sample from high-dimensional, complex distributions.

To set the scene, let's take the simplest approach to generative modelling that has nothing to do with neural networks. Let's imagine 
we are provided with $N_s$ multi-dimensional arrays and we are told that they come from a multi-variate gaussian distribution. We can 
set up a generative modelling task as follows:

- Training
     - Compute the sample mean and covariance from the training samples: $\boldsymbol \mu, \boldsymbol \Sigma$
     - Apply the Cholesky decomposition to the covariance matrix: $\boldsymbol \Sigma = \mathbf{L} \mathbf{L}^T$
  
- Inference / Generation
     - Sample a vector from a unitary, zero-mean normal distribution $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
     - Create a new sample from the true distribution: $\tilde{\mathbf{x}} = \mathbf{L} \mathbf{z} + \boldsymbol \mu$

Unfortunately, multi-dimensional distributions that we usually find in nature are hardly gaussian and this kind of simple
generative modelling procedure falls short. Nevertheless, the approach that we take with some of the more advanced generative modelling
methods that we are going to discuss later on in this lecture does not differ from what we have done so far. A training phase, where the 
free-parameters of the chosen parametric model (e.g., a NN) are learned from the available data, followed by a generation phase that uses
the trained model and some stochastic input (like the $\mathbf{z}$ vector in the example above).

## Variational AutoEncoders (VAEs)

As the name implies, these networks take inspiration from the AutoEncoder networks that we have presented in the previous lecture. However, some 
small, yet fundamental changes are implemented to the network architecture as well as the learning process (i.e., loss function) to turn such family
of networks from being able to perform dimensionality reduction to being generative models. 

Let's start by looking at a schematic representation of a VAEs:

![VAE](figs/vae.png)

Even before we delve into the mathematical details, we can clearly see that one main change has been implemented to the network architecture:
instead of directly producing a vector $\mathbf{z} \in \mathbb{R}^{N_l}$, the encoder's output is composed of two vectors 
$\boldsymbol \mu \in \mathbb{R}^{N_l}$ and $\boldsymbol \sigma \in \mathbb{R}^{N_l}$ that represent the mean and standard deviation of a $N_l$ dimensional
gaussian distribution (with uncorrelated variables, i.e., diagonal covariance matrix). These two vectors are fed together to a sampler,
who similar to what we did before, produces a sample from the following gaussian distribution: $\mathcal{N}(\boldsymbol \mu, diag\{ \boldsymbol \sigma \})$.
In practice this is however achieved by sampling a vector and then transforming it into the desired distribution, 
$\mathbf{z} = \boldsymbol \sigma \cdot \mathbf{z} + \boldsymbol \mu$ where $\cdot$ refers to an element-wise product. 

### Reparametrization trick

This rather simple trick is referred
to as *Reparametrization trick* and it is stricly needed in neural networks every time we want to introduce a stochastic procees within the computational graph.
In fact, by simply having a stochastic process parametrized by a certain mean and standard deviation that may come from a previous part of the computational graph
(as in VAEs) we lose the possibility to perform backpropagation. Instead if we decouple the stochastic component (which we are not interested to update, and 
therefore to backpropagate onto) and the deterministic component(s), we do not lose access to backpropagation:

![REPARAMETRIZATIONTRICK](figs/reptrick.png)

### Why VAEs?

Before we progress in discussing the loss function and training procedure of VAEs, a rather simple question may arise: 'Why can we not use AEs for
generative modelling?'

In fact, this could be achieved by simply modifying the inference step:

![GENAE](figs/generativeae.png)

where instead of taking a precomputed $\mathbf{z}$ vector (from a previous stage of compression), we could sample a new $\mathbf{z}$ 
value from a properly crafted distribution (perhaps chosen from statistical analysis of the training latent vectors) at any time we want 
to create a new sample.

Unfortunately, whilst this idea may sound reasonable, we will be soon faced with a problem. In fact, the latent manifold learned by a AE may
not be regular, or in other words it may be hard to ensure that areas of such manifold that have not been properly sampled by the training data will
produce meaningful samples $\tilde{\mathbf{z}}$. Just to give an idea, let's look at the following schematic representation:

![LATENTAE](figs/latentspaceae.png)

as we can see, if a part of the latent 1-d manifold is not rich in training data, the resulting generated sample may be non-representative at all.
Whilst we discussed techniques that can mitigate this form of overfitting (e.g., sparse AEs), VAEs bring the learning process to a whole new level
by choosing a more appropriate regularization term $R(\mathbf{x}^{(i)} ;\theta,\phi)$ to add to the reconstruction loss.

### Regularization in VAEs

In order to better understand the regularization choice in VAEs, let's look once again at a schematic representation of VAEs but this time in a
probabilistic mindset:

![VAEPROB](figs/vaeprob.png)

where we highlight here the fact that the encoder and decoder can be seen as probability approximators. More specifically:

- $e_\theta(\mathbf{x}) \approx p(\mathbf{z}|\mathbf{x})$: the encoder learns to sample from the latent space distribution conditioned on a specific input
- $d_\phi(\mathbf{z}) \approx p(\mathbf{x}|\mathbf{z})$: the decoder learns to sample from the true distribution conditioned on a specific latent sample

By doing so, we can reinterpret the reconstruction loss as the negative log-likelihood of the decoder. And, provided that we have defined a 
prior for the latent space $\mathbf{z} \sim P(\mathbf{z})$, we can learn the parameters of the decoder by ensuring that the posterior does not deviate
too much from the prior. This can be achieved by choosing:

$$
R(\mathbf{x} ;\theta,\phi) = KL(p(\mathbf{z}|\mathbf{x})||p(\mathbf{z}))
$$

As in any statistical learning process, the overall loss of our VAEs shows a trade-off between the likelihood (i.e., learning from data) and 
prior (i.e., keeping close to the initial guess).


## Additional readings

- The flow of this lecture is heavily inspired by this [blog post](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73)
- A Python library that can help you step up your game with Variational Inference is [Pyro](https://pyro.ai) from Uber.