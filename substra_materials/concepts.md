## Main concepts


Connect is an application to train and evaluate machine learning models on distributed datasets without centralising the data and more generally without compromising the privacy of the data. Connect keeps a traceability of operations done to train and evaluate models.


It supports simple training and evaluation scheme, such as training a model on data in a center A and evaluating the model on data in a center B, or more complex federated learning (FL) schemes, such as Federated Averaging. It also enables to do both horizontal FL (same features across participants) and vertical FL (different features across participants), as well as multi-partners multi-task learning (multi tasks that can be different from one participant to another).


To make this possible, 4 main assets are defined:


* Metrics
* Datasets and Data Samples
* Algorithms
* Models and Compute Plans


### Metrics
A metric is used to evaluate a model performance. Concretely, a metric corresponds to an - archive (tar or zip file) containing
* Python scripts which code for the metric computation
* a Dockerfile which contains the required dependencies of the Python scripts


### Dataset:
A dataset represents the data in Connect. It is made up of:
* an opener, which is a script used to load the data from files into memory.
* Data samples: a folder containing the data files


### Algorithms
An algorithm specifies the method to train a model on a dataset. It specifies the model type and architecture, the loss function, the optimizer, hyperparameters and, also identifies the parameters that are tuned during training. Concretely, an algorithm corresponds to an archive (tar or zip file) containing:
* Python scripts which code for the algorithm. Importantly, a train and a predict functions have to be defined 
* a Dockerfile which contains the required dependencies of the Python scripts


There are three types of algorithms:


* Simple algorithm:
This algorithm can be used with tasks of kind train tuple and produces a single model.
* Composite algorithm:
This kind of algorithm makes it possible to train a trunk and head model:
* the trunk model can be shared among all nodes
* the head model remains private to the node where it was trained
This enables vertical FL, as well as multi-partners multi-task FL.
* Aggregate algorithm
This algorithm type is used to aggregate models or model updates. An aggregate algorithm does not need data to be used.


### Models/Model updates
A Model or a Model Update is a potentially large file containing the parameters or update of parameters of a trained model[a]. In the case of a neural network, a model would contain the weights of the connections. It is either the result of training an Algorithm with a given Dataset, corresponding to a training task (Train tuple or Composite train tuple); or the result of an Aggregate Algorithm aggregating models or model updates; corresponding to an aggregation task (Aggregate tuple).




### Compute plans and tasks
Compute plan
A set of training (Train tuple or Composite train tuple), aggregation (Aggregate tuple) and testing tasks (Test tuple) gathered together towards building a final model.
The user should register a compute plan rather than separate tasks. 


#### Train tuple
The specification of a training task of a simple algorithm on a dataset potentially using input models or model updates[b]. It leads to the creation of a model or model update.


#### Composite train tuple
The specification of a training task of a composite algorithm on a dataset potentially using input trunk and head models or model updates. It leads to the creation of a trunk and head model or model update. Depending on associated permissions, a trunk model or model update can be shared with other nodes, whereas a head model remains in the node where it was created.


#### Aggregate tuple
The specification of an aggregation task of several models or model updates using an aggregate algo. It leads to the creation of one model or model update.


#### Test tuple
The specification of a testing task of a model. It evaluates the performance of the model using one or several metrics with a dataset.
