# More on Neural Networks

In this lecture, we will delve into some more advanced topics associated to the creation and training of deep 
neural networks.

## Backpropagation

First of all, once a neural network architecture has been defined for the problem at and, we need a method
that can learn the best set of free parameters of such nonlinear function represented as $f_\theta$.

More specifically, we want to initialize the network with some random weights and biases (we will soon discuss how
such initialization can be performed) and use the training data at hand to improve our weights and biases in order
to minimize a certain loss function. Whilst this can be easily done by means of gradient based optimizers like those
presented in Lecture 3, a key ingredient that we need to provide to such algorithms is represented by the gradient
of the loss function with respect to each and every weight and bias paramtets. 

We have already alluded at a technique that can do so whilst discussing a simple logistic regression model. This
is generally referred to by the ML community as *back-propagation* and more broadly by the mathematical community
as *Reverse Automatic Differentiation*. Let's start by taking the same schematic diagram used for the logistic regression
example and generalize it to a N-layer NN:

![BACKPROP_NN](figs/backprop_nn.png)

The main change here, which we will need to discuss in details, is the fact that in the forward pass 
we feed the input into a stack of linear layers prior to computing the loss function. The backpropagation
does need to be able to keep track of the chain of operations (i.e., computational graph) and traverse it
back. However, as already done for the logistic regression model, all we need to do is to write the entire
chain of operations as a chain of atomic ones that we can then easily traverse back. Let's do this for
the network above and a single training sample $\textbf{x}$:

$$
\textbf{z}^{[1]} = \textbf{W}^{[1]}\textbf{x} + \textbf{b}^{[1]}, \quad
\textbf{a}^{[1]} = \sigma(z^{[1]}),
$$

$$
\textbf{z}^{[2]} = \textbf{W}^{[2]}\textbf{a}^{[1]} + \textbf{b}^{[2]}, \quad
\textbf{a}^{[2]} = \sigma(z^{[2]}),
$$

$$
\textbf{z}^{[3]} = \textbf{W}^{[3]}\textbf{a}^{[2]} + \textbf{b}^{[3]}, \quad
a^{[3]} = \sigma(z^{[3]}),
$$

$$
l = \mathscr{L}(y,a^{[3]}).
$$

Given such a chain of operations, we are now able to find the derivatives of the loss function with
respect to any of the weights or biases. As an example we consider here $\partial l / \partial \textbf{W}^{[2]}$:

$$
\frac{\partial l}{\partial \textbf{W}^{[2]}} = \frac{\partial l}{\partial a^{[3]}} \frac{\partial a^{[3]}}{\partial \textbf{z}^{[3]}}
\frac{\partial \textbf{z}^{[3]}}{\partial \textbf{a}^{[2]}} \frac{\partial \textbf{a}^{[2]}}{\partial \textbf{z}^{[2]}} 
\frac{\partial \textbf{z}^{[2]}}{\partial \textbf{W}^{[2]}}
$$

Assuming for simplicity that the binary cross-entropy and sigmoid functions are used here as loss and activation functions, respectively:

$$
\frac{\partial l}{\partial a^{[3]}} \frac{\partial a^{[3]}}{\partial z^{[3]}} = a^{[3]} - y
$$

$$
\frac{\partial z^{[3]}}{\partial \textbf{a}^{[2]}} = \textbf{W}^{[3]}
$$

$$
\frac{\partial \textbf{a}^{[2]}}{\partial \textbf{z}^{[2]}} = \textbf{a}^{[2]}(1-\textbf{a}^{[2]})
$$

$$
\frac{\partial \textbf{z}^{[2]}}{\partial \textbf{W}^{[2]}} = \textbf{a}^{[1]}
$$

which put together:

$$
\frac{\partial l}{\partial \textbf{W}^{[2]}} = [(\textbf{a}^{[2]}(1-\textbf{a}^{[2]})) \cdot \textbf{W}^{[3]T}(a^{[3]} - y)] \textbf{a}^{[1]T}
$$

where $\cdot$ is used to refer to element-wise products. Similar results can be obtained for the bias vector
and for both weights and biases in the other layers as depicted in the figure below for a 2-layer NN:

![BACKPROP_NN1](figs/backprop_nn1.png)

To conclude, the backpropagation equations in the diagram above are now generalized for the case 
of $N_s$ training samples $\textbf{X} \in \mathbb{R}^{N \times N_s}$ and a generic activation function
$\sigma$ whose derivative is denoted as $\sigma'$. Here we still assume an output
of dimensionality one -- $\textbf{Y} \in \mathbb{R}^{1 \times N_s}$:

$$
\textbf{dZ}^{[2]}=\textbf{A}^{[2]}-\textbf{Y} \qquad (\textbf{A}^{[2]},\textbf{dZ}^{[2]} \in \mathbb{R}^{1 \times N_s})
$$

$$
\textbf{dW}^{[2]}= \frac{1}{N_s} \textbf{dZ}^{[2]}\textbf{A}^{[1]T} \qquad (\textbf{A}^{[1]} \in \mathbb{R}^{N^{[1]} \times N_s})
$$

$$
db^{[2]}= \frac{1}{N_s} \sum_i \textbf{dZ}_{:,i}^{[2]}
$$

$$
\textbf{dZ}^{[1]}=\textbf{W}^{[2]^T}\textbf{dZ}^{[2]} \cdot \sigma'(\textbf{Z}^{[1]})  \qquad (\textbf{dZ}^{[1]} \in \mathbb{R}^{N^{[1]} \times N_s})
$$

$$
\textbf{dW}^{[1]}= \frac{1}{N_s} \textbf{dZ}^{[1]}\textbf{X}^T
$$

$$
\textbf{db}^{[1]}= \frac{1}{N_s} \sum_i \textbf{dZ}_{:,i}^{[1]}
$$

## Initialization
Neural networks are hihgly nonlinear functions. The associated cost function used in the training
process in order to optimize the network weights and biases is therefore non-convex and contains
several local minima and saddle points.

A key component in non-convex optimization is represented by the starting guess of the parameters
to optimize, which in the context of deep learning is identified by initialization of weights and biases.
Whilst a proper initialization has been shown to be key to a succesful training of deep train NNs, 
this is a very active area of research as initialization strategies are so far mostly based on heuristic 
arguments and experience.

### Zero initialization
First of all, let's highlight a bad initialization choice that can compromise the training no matter the 
architecture of the network and other hyperparamters. A common choice in standard optimization in the absence
of any strong prior information is to initalize all the paramters to zero: if we decide to follow such a strategy
when training a NN, we will soon realize that training is stagnant due to the so called *symmetry problem*
(also referred to as *symmetric gradients*). Note that a similar situation arises also if we 
choose a constant values for weights and biases (e.g., $c^{[1]}$ for all the weights and biases in the first layer and 
$c^{[2]}$ for all the weights and biases in the second layer):

Let's take a look at this with an example:

![ZEROINIT](figs/zeroinit.png)

Since the activations are constant vectors, back-propagation produces constant updates for the weights (and biases),
leading to weights and biases to never lose the initial symmetry.

### Random initialization
A more appropriate way to initialize the weights of a neural network is to sample their
values from random distributions, for example:
$$
w_{ij}^{[.]} \sim \mathcal{N}(0, 0.01)
$$
where the choice of the variance is based on the following trade-off: too small variance leads to the 
vanishing gradient problem (i.e., slow training), whilst too high variance leads to the 
exploding gradient problem (i.e., unstable training). On the other hand, for the biases we can use zero or a constant value. If you remember, we have already
mentioned this when discussing the ReLU activation function: a good strategy to limit the amount of
negative values as input to this activation function is to choose a small constant bias (e.g., $b=0.1$).

Whilst this approach provides a good starting point for stable training of neural networks, more advanced
initialization strategies have been proposed in the literature:

- **Uniform**: the weights are initialized with uniform distributions whose variance depend on the
  number of units in the layer:
  $$
  w_{ij}^{[k]} \sim \mathcal{U}(-1/\sqrt{N^{[k]}}, 1/\sqrt{N^{[k]}})
  $$
  or
  $$
  w_{ij}^{[k]} \sim \mathcal{U}(-\sqrt{6/(N^{[k-1]}+N^{[k]})}, \sqrt{6/(N^{[k-1]}+N^{[k]})})
  $$
  This strategy is commonly used with FC layers.
  

- **Xavier**: the weights are initialized with normal distributions whose variance depend on the
  number of units in the layer:
  $$
  w_{ij}^{[k]} \sim \mathcal{N}(0, 1/N^{[k]})
  $$
  This strategy ensures that the variance remains the same across the layers. Xavier initialization
  is very popular especially in layers using Tanh activations.
  
- **He**: the weights are initialized with normal distributions whose variance depend on the
  number of units in the layer:
  $$
  w_{ij}^{[k]} \sim \mathcal{N}(0, 2/N^{[k]})
  $$
  This strategy ensures that the variance remains the same across the layers. Xavier initialization
  is very popular especially in layers using ReLU activations.
  
  
Finally, if you are interest to learn more about initialization I reccomend reading (and reproducing)
the following blog posts: [1](https://medium.com/@safrin1128/weight-initialization-in-neural-network-inspired-by-andrew-ng-e0066dc4a566)
and [2](https://www.deeplearning.ai/ai-notes/initialization/).


WHY NN/DEEP LEARNING TOOK OFF
-----------------------------

Finally lets remark why theories are all from '80 but until early 2000 NN were not popular (niche field)

"The core ideas behind modern feedforward networks have not changed sub-stantially since the 1980s. 
The same back-propagation algorithm and the same approaches to gradient descent are still in use. Most of 
the improvement in neuralnetwork performance from 1986 to 2015 can be attributed to two factors. First,larger 
datasets have reduced the degree to which statistical generalization is achallenge for neural networks. Second, neural 
networks have become much larger,because of more powerful computers and better software infrastructure"

Plus a few algorithmic changes:
- "One of these algorithmic changes was the replacement of mean squared errorwith the cross-entropy family of loss functions. 
Mean squared error was popular inthe 1980s and 1990s but was gradually replaced by cross-entropy losses and the principle 
of maximum likelihood as ideas spread between the statistics community and the machine learning community. 
The use of cross-entropy losses greatly improved the performance of models with sigmoid and softmax outputs, whichhad previously 
Suﬀered from saturation and slow learning when using the meansquared error loss" --> NOTE for regression ML with gaussianity assumption
on p(y|x) is still MSE, but only for this edge case!

- "change that has greatly improved the performanceof feedforward networks was the replacement of sigmoid hidden units 
  with piecewiselinear hidden units, such as rectiﬁed linear units. Rectiﬁcation using themax{0, z}function was 
  introduced in early neural network models and dates back at least as faras the cognitron and neocognitron (Fukushima, 1975, 1980)...
  This began to change in about 2009. Jarrett et al. (2009)observed that “using a rectifying nonlinearity is the single most 
  important factorin improving the performance of a recognition system,” --> LEARN TO CHALLANGE STATUS QUO, SOMETIMES UNDERSTANDING OF
  PROBLEMS CAN CHANGE OR NEW EXTERNAL FACTORS (EG MORE DATA) MAKE SOMETHING THAT WAS WORSE BECOME BETTER... SIMILAR STORY IN FWI FOR GEOPHYSCISTS
  

Also add mixture density 'network' (example of petroelastic with facies - Andrews paper) - https://towardsdatascience.com/a-hitchhikers-guide-to-mixture-density-networks-76b435826cca
