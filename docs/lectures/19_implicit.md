# Implicit neural networks

Neural networks consists of a sequence of consecutive operations that are typically defined explicitly. An explicit operation is one that computes the output directly
from a sequence of explicit operations applied to the input. A simple example is a feed-forward MLP, where the transition from one layer to the next is done by the following sequence 
of operations

$$
\begin{aligned}
z_i & = W_iz_{i-1} + b_i \\
a_i & = \sigma_i(z_i)
\end{aligned}
$$

Additionally one could add operations like batch normalization and max pooling, all of which are given explicitly. Alternatively, two variables can be related
via an implicit equation. A simple example of an explicit function versus an implicit function is $y = x^2$ versus $x^2 + y^2 = 1$. From the second example it 
becomes clear why implicit functions are sometimes favorable, since the implicit function $x^2 + y^2 = 1$ has an explicit counterpart with two equations, namely
$y = \sqrt{1 - x^2}$ and $y = -\sqrt{1 - x^2}$. In a more abstract fashion, we can write an explicit equation as

$$
  y = f(x),
$$

and an implicit equation as

$$
  f(x, y) = 0.
$$

Neural networks can be defined implicitly as well through the concept of implicit layers and were introduced under the name *Deep Equilibrium Models (DEQ)*. This concept 
is a bit abstract but the nice thing about this paradigm is that the memory requirements for deep networks are constant. To understand this concept we need to cover
two fundamental concepts:

- Implicit functions and the implicit function theorem. Taking derivatives of explicit functions is easy, since we have an explicit relation of the output with respect
to the input, and we can compute $\frac{dy}{dx}$ in a straightforward manner. However, if $y$ is only given through $f(x, y) = 0$ then computing the derivative is 
less straightforward.
- Fixed point iterations. Fixed point iterations are iterations of the form $x_{k+1} = \mathcal{F}(x_k)$ and we call a vector $x_{\star}$ a *fixed point of $\mathcal{F}$* 
if $x_{\star} = \mathcal{F}(x_{\star})$. DEQs are based on the idea that the layers of a neural network will eventually reach a fixed point.

## Fixed point iterations
Consider the following fixed point iteration

$$
  z_{k+1} = \tanh(Wz_k + b + x).
$$

This is essentially repeated application of one layer of a neural network with weight matrix $W$, bias $b$, some input $x$ and activation function $\tanh$. Assuming for now that 
a fixed point actually exists we iterate until convergence, i.e. $z_{\star} = \mathcal{F}(z_{\star})$ up to some tolerance. Alternatively, we can write the above equation as 

$$
  z - \tanh(Wz + b + x) = 0,
$$

where the function is now implicitly defined. Defining

$$
  g(x, z) := z - \tanh(Wz + b + x),
$$

the goal is now to solve the root finding problem

$$
  g(x, z_{\star}(x)) = 0,
$$

where $z_{\star}(x)$ denotes the solution depending on $x$. Let the solution to this problem be given by $z_{\star}(x)$ and assume we want 
to compute $\frac{dz_{\star}(x)}{dx}$ (note that we could choose to differentiate through any parameter, for example the weight matrix, 
but this is just for illustrative purposes). Since we only have access to $z_{\star}$ through the equation $g(x, z_{\star}(x)) = 0$ need to 
differentiate through this equation to obtain $\frac{dz_{\star}(x)}{dx}$. This yields:

$$
  \frac{\partial}{\partial x}g(x, z_{\star}(x)) = \frac{\partial g(x, z_{\star})}{\partial x} +  \frac{\partial g(x, z_{\star})}{\partial z_{\star}}\frac{\partial z_{\star}(x)}{\partial x} = 0
$$

This equation allows us to solve for $\frac{dz_{\star}(x)}{dx}$ as follows

$$
  \frac{\partial z_{\star}(x)}{\partial x} = -\left(\frac{\partial g(x, z_{\star})}{\partial z_{\star}}\right)^{-1}\frac{\partial g(x, z_{\star})}{\partial x}
$$

The main question here is whether existence is guaranteed. The *implicit function theorem* states that if a fixed point exists and the function 
$g$ is differentiable with non-singular Jacobian around $z_{\star}$ there exists a unique function $z_{\star}(x)$. The key point here is that 
one can differentiate through $z_{\star}$ without needing to differentiate through the solver used to obtain the fixed point. This saves a huge amount of 
memory that would otherwise be needed in order to perform backpropagation. 

This observation has led to the development of the *Deep Equilibrium Network*. This network has the following structure:

$$
\begin{aligned}
z_1 & = 0 \\
z_i & = \sigma_i(Wz_i + Ux + b_i), \quad i=1,\ldots, k \\
h(x) & = W_kz_k + b_k
\end{aligned}
$$

As we can see, DEQs apply a fixed point iteration to a single layer of a neural network. The question is whether this fixed point iteration
actually converges: It could also blow-up or oscillate. It turns out that in general the fixed point iteration converges. As you can probably guess
at this point, the fixed point iteration is solved using implicit differentiation, thereby bypassing the need to store any information necessary for 
the backward pass. This way one can build an extremely deep network. If we now want to update the weights of the neural network we need to evaluate the 
partial derivative with respect to $W$. Given that $z_{\star}$ is a fixed point we have

$$
  z_{\star} = f(x, z_{\star}) \: \Leftrightarrow \: \frac{\partial z_{\star}}{\partial W} = \frac{\partial f(x, z_{\star})}{\partial W}
$$

Computing $\frac{\partial f(x, z_{\star})}{\partial W}$ via implicit differentiation and rearranging terms gives

$$
  \frac{\partial z_{\star}}{\partial W} =  \left(I - \frac{\partial f(x, z_{\star})}{\partial z_{\star}}\right)^{-1}\frac{\partial f(x, z_{\star})}{\partial W}
$$

Backpropagation actually implements the transpose of this expression, i.e.:

$$
  \left(\frac{\partial z_{\star}}{\partial W}\right)^Ty =  \left(\frac{\partial f(x, z_{\star})}{\partial W}\right)^T\left(I - \frac{\partial f(x, z_{\star})}{\partial z_{\star}}\right)^{-T}y,
$$

where $y$ is some vector we apply the gradient to. Evaluating the gradient is now a two-step process:

- Evaluate $\left(I - \frac{\partial f(x, z_{\star})}{\partial z_{\star}}\right)^{-1}y$. Since this matrix tends to be large we do not evaluate
the inverse directly, but rather solve the linear system 
$$
  y = \left(I - \frac{\partial f(x, z_{\star})}{\partial z_{\star}}\right)g \quad \Leftrightarrow \quad g = \frac{\partial f(x, z_{\star})}{\partial z_{\star}}g + y.
$$
- Compute 
$$
  \left(\frac{\partial z_{\star}}{\partial W}\right)^Ty = \left(\frac{\partial f(x, z_{\star})}{\partial W}\right)^Tg
$$

So far we have considered a rather simple model for the DEQ. We have assumed a constant weight $W$ accross the layers and have assumed a simple
feed-forward model. It turns out that a feed-forward neural network with constant weights accross the layers is actually equivalent to a neural network
with a layer-dependent matrix, which is summarized in the following theorem by [Bai et al., 2019](https://arxiv.org/pdf/1909.01377.pdf):

Consider a traditional $L$-layer MLP

$$
  z_{i+1} = \sigma_{i}(W_iz_i + b_i), \quad i=0,\ldots,L-1, \quad z_0 = x.
$$

This network is equivalent to the following weight-tied network of equivalent depth:

$$
  \tilde{z}_{i+1} = \tilde{\sigma}(W_zz_i + \tilde{b} + Ux), \quad i=0, \ldots, L-1, \quad \tilde{z}_{0} = (0, \ldots, 0)^T
$$

We prove the theorem for the case $L = 4$, but it extends to general $L$. Define the matrices

$$
  W_z = \begin{bmatrix} 0 & 0 & 0 & 0 \\ W_1 & 0  & 0 & 0 \\ 0 & W_2 & 0 & 0 \\ 0 & 0 & W_{3} & 0 \end{bmatrix}, \:
U = \begin{bmatrix} W_0 \\ 0 \\ 0 \\ 0 \end{bmatrix}, \: \tilde{b} = \begin{bmatrix} b_0 \\ b_1 \\ b_2 \\ b_3 \end{bmatrix}, \: \tilde{\sigma} = \begin{bmatrix} \sigma_0 \\ \sigma_1 \\ \sigma_2 \\ \sigma_3 \end{bmatrix}.
$$

Then after one iteration we have

$$
  \tilde{z}_1 = \tilde{\sigma}(W_z\tilde{z}_0 + Ux + \tilde{b}) = \begin{bmatrix} \sigma_0(W_0x + b_0) \\ \sigma_1(b_1) \\ \sigma_2(b_2) \\ \sigma_3(b_3) \end{bmatrix} 
= \begin{bmatrix} z_0 \\ \sigma_1(b_1) \\ \sigma_2(b_2) \\ \sigma_3(b_3) \end{bmatrix}.
$$

For the second iteration we have

$$
  W_z\tilde{z_1} = \begin{bmatrix} 0 & 0 & 0 & 0 \\ W_1 & 0  & 0 & 0 \\ 0 & W_2 & 0 & 0 \\ 0 & 0 & W_{3} & 0 \end{bmatrix}\begin{bmatrix} z_0 \\ \sigma_1(b_1) \\ \sigma_2(b_2) \\ \sigma_3(b_3) \end{bmatrix} = \begin{bmatrix} 0 \\ W_1z_0 \\ W_2\sigma_2(b_2) \\ W_3\sigma_2(b_2) \end{bmatrix}
$$

and hence

$$
  \tilde{z}_{2} = \tilde{\sigma}(W_zz_1 + \tilde{b} + Ux) = \begin{bmatrix} \sigma_0(W_0x + b_0) \\ \sigma_1(W_1z_0 + b_1) \\ \sigma_2(W_2\sigma_1(b_1) + b_2) \\ \sigma_3(W_3\sigma_2(b_2) + b_3) \end{bmatrix} 
= \begin{bmatrix} z_0 \\ z_1 \\ \sigma_2(W_2\sigma_1(b_1) + b_2) \\ \sigma_3(W_3\sigma_2(b_2) + b_3) \end{bmatrix}. 
$$

Similarly, for the next layer we obtain

$$
  \begin{bmatrix} 0 & 0 & 0 & 0 \\ W_1 & 0  & 0 & 0 \\ 0 & W_2 & 0 & 0 \\ 0 & 0 & W_{3} & 0 \end{bmatrix}\begin{bmatrix} z_0 \\ z_1 \\ \sigma_2(W_2\sigma_1(b_1) + b_2) \\ \sigma_3(W_3\sigma_2(b_2) + b_3) \end{bmatrix} = \begin{bmatrix} 0 \\ W_1z_0 \\ W_2z_1 \\ W_3\sigma_2(W_2\sigma_1(b_1) + b_2) \end{bmatrix}
$$

which leads to 

$$
  \tilde{z}_3 = \begin{bmatrix} \sigma_0(W_0x + b_0) \\ \sigma_1(W_1z_0 + b_1) \\ \sigma_2(W_2z_1 + b_2) \\ \sigma_3(W_3\sigma_2(W_2\sigma_1(b_1) + b_2) + b_3) \end{bmatrix} = \begin{bmatrix} z_1 \\ z_2 \\ z_3 \\ \sigma_3(W_3\sigma_2(W_2\sigma_1(b_1) + b_2) + b_3) \end{bmatrix}
$$

Then, finally,

$$
  \begin{bmatrix} 0 & 0 & 0 & 0 \\ W_1 & 0  & 0 & 0 \\ 0 & W_2 & 0 & 0 \\ 0 & 0 & W_{3} & 0 \end{bmatrix}\begin{bmatrix} z_1 \\ z_2 \\ z_3 \\ \sigma_3(W_3\sigma_2(W_2\sigma_1(b_1) + b_2) + b_3) \end{bmatrix} = \begin{bmatrix} 0 \\ W_1z_1 \\ W_2z_2 \\ W_3z_3 \end{bmatrix}
$$

and hence

$$
  \tilde{z}_4 = \begin{bmatrix} \sigma_0(W_0x + b_0) \\ \sigma_1(W_1z_0 + b_1) \\ \sigma_2(W_2z_1 + b_2) \\ \sigma_3(W_3z_2 + b_3) \end{bmatrix} = \begin{bmatrix} z_1 \\ z_2 \\ z_3 \\ z_4 \end{bmatrix}.
$$

Moreover, note that we have only used a single layer DEQ as opposed to the multi-layer architecture that is typical for powerful neural networks. However,
any deep neural network can be represented as a deep neural network. The argument is as follows. Assume that construct a two-layer network $g_2(g_1(x))$. This can 
be posed a single layer DEQ using the following relation:

$$
  f(z, x) = f\left( \begin{bmatrix} z_1 \\ z_2 \end{bmatrix}, x\right) = \begin{bmatrix} g(x) \\ g(z_1) \end{bmatrix}
$$

That is, the complexity of the extra layer can simply be added by concatenating the two layers to make a single layer neural network. The same argument
holds for stacking DEQs: a single DEQ can model any number of stacked DEQs.

Finally, we can increase the complexity of the DEQ by substituting the simple feed-forward neural network with any sequence of operations, including
convolutions, normalizations, grouping and skip connections. 


## Further reading
These notes are essentially a summary of the following tutorial, specifically chapters 1 and 4:

- [Implicit layer tutorial](http://implicit-layers-tutorial.org/)

Below is the paper introducing Deep Equilibrium Models (DEQ):

- [Bai et al., 2019](https://arxiv.org/pdf/1909.01377.pdf)
