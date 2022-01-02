# Machine Learning Experimentation tracking with WANDB

A good idea when experimenting with Machine Learning models is to keep track of the parameters used in 
each experiment as well as the evaluation metrics. Failing to do so will result in a lot of wasted time
repeating experiments and not being able to objectively understand which parameters have more or less
influence on the final results.

A great tool for this is called [WANDB](https://wandb.ai/site). It allows making minimal changes to your
code to keep track of the input parameters that we are interested to change and the metrics we want to 
assess.

## Set up

To begin with, head over to the WANDB site and create an Account. Once the account is created you will be added to the ``mlgeoscience`` team.
At this point head over to https://wandb.ai/settings and set ``Default location to create new projects`` 
as ``mlgeoscience``.

On your terminal, run
```
wandb login
```

You should see that you are authenticated as ``mlgeoscience`` user.

Read this page for more details: https://docs.wandb.ai/quickstart.


## Running an experiment

We have created a simple example in the `logreg.py` script, where Logistic Regression is applied to the 
Two-Moon datasets. In this case we focus on 3 parameters:

- hidden layer size
- learning rate
- number of epochs

and we monitor two metrics:

- training accuracy
- validation accuracy


We can run an experiment as follows:

```
python logreg.py log-regression -e 1000 -l 1. -u 20
```

Your experiment will run and logged by Wandb. Head over to your Wandb page and check out this experiment
and compare with all the other you have previously run.


## Running a sweep

Even more interesting, WandDB can help making your hyperparameter search easy and trackable.
Sometimes when you start a new ML project you quickly end up with a series of hyperparameters to choose,
which most often than not you are not really able to select without some trial and error. Do not spend
time yourself running 10s or 100s of examples, let WandDB do it for you. In the webpage, click the Sweep icon,
