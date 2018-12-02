# SuperLearner
Sklearn based Super Learning Stacked model

Stacking is a broad class of algorithms that involves training a second-level "metalearner" to ensemble a group of base learners. The type of ensemble learning implemented in H2O is called "super learning", "stacked regression" or "stacking." Unlike bagging and boosting, the goal in stacking is to ensemble strong, diverse sets of learners together.

## Super Learner Algorithm
Here is an outline of the tasks involved in training and testing a Super Learner ensemble.

### Set up the ensemble
* Specify a list of L base algorithms (with a specific set of model parameters).
* Specify a metalearning algorithm.

### Train the ensemble
* Train each of the L base algorithms on the training set.
* Perform k-fold cross-validation on each of these learners and collect the cross-validated predicted values from each of the L algorithms.
* The N cross-validated predicted values from each of the L algorithms can be combined to form a new N x L matrix. This matrix, along wtih the original response vector, is called the "level-one" data. (N = number of rows in the training set)
* Train the metalearning algorithm on the level-one data.
* The "ensemble model" consists of the L base learning models and the metalearning model, which can then be used to generate predictions on a test set.

### Predict on new data
* To generate ensemble predictions, first generate predictions from the base learners.
* Feed those predictions into the metalearner to generate the ensemble prediction.


## Execute

```python src/stacked_learner.py```


## Reference
* http://docs.h2o.ai/h2o-tutorials/latest-stable/tutorials/ensembles-stacking/index.html
