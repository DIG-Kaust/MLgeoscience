# Introduction to Machine Learning

Humans have long dreamed of creating *machines that can think and act independently*. For many years this has been the
aim of **Artificial Intelligence (AI)**. In the early days of AI, many problems that are difficult to solve by humans 
(e.g., large summations or multiplications, solution of system of equations) turn out to be easier for computers as long
as humans could define a list of tasks that the machines could perform at faster speed and higher precisions than humans
can do themselves.

On the other hand, tasks that are very easily solved by adult humans and even kids (e.g., recognizing animals in pictures
or singing a song) turned out to be very difficult for computers. The main reason of such difficulties lies in the fact
that humans cannot explain in words (and with a simple set of instructions) how they have learned to accomplish these tasks.
This is where instead the second era of AI solutions, belonging to the field of **Machine Learning (ML)**, have shown astonishing
results in the last decade. Instead of relying on hard-coded rules, these algorithms operate in a similar fashion to human 
beings as they learn from *experience*. In other words, given enough *training data* in the form of inputs (e.g., photos)
and outputs (e.g., label of the animal present in the photo), ML algorithms can learn a complex nonlinear mapping between
them such that they can infer the output from the input when provided with unseen inputs. 

A large variety of ML algorithms have been developed by the scientific community, ranging from the basic *linear* and *logistic 
regression* that we will see in our [third lecture](lectures/linreg.md), decision tree-based statistical methods such 
as *random forrest* or *gradient boosting*, all the way to *deep neural networks*, which have recently
shown to outperform previously developed algorithms in many fields (e.g., computer science, text analysis and speech recognition,
seismic interpretation). This subfield has grown exponentially in the last few years and it is now referred to as **Deep Learning**
and will be subject of most of our course. Whilst sharing the same underlying principle of learning from experience in the form 
of a training data, different algorithms presents their own strenght and limitations and a machine learning practitioner 
must make a careful judgment at any time depending on the problem to be solved.

**FIGURE**

Machine Learning is divided into 2 main categories:

- Supervised Learning: learn a function that maps an input to an output ($X \rightarrow Y$). Inputs are also referred to as
  features and output are called targets. In practice we have access to a number of training pairs 
  $\{ \textbf{x}_i, \textbf{y}_i \} \; i=1,..,N$ and we learn $\textbf{y}_i=f_\theta(\textbf{x}_i)$ 
  where $f_\theta$ is for example parametrized via a neural network. Two main applications
  of supervised learning are
    * *Classification*: the target is discrete
    * *Regression*: the target is continuous
  
- Unsupervised Learning: learn patterns from unlabelled data. These methods have been shown to be able to find compact 
  internal representation of the manifold the input data belongs to. Such compact representations can become valuable
  input features for subsequent tasks of supervised learning. We will get into the mathematical details of this family of
  models in the second part of our course.
  
Input data can also come in 2 different types:

- Structured data: tables (e.g., databases)
- Unstructured data: images, audio, text, ...