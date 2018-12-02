import sklearn
import numpy as np

class SuperSklearn(sklearn.base.BaseEstimator):
    def __init__(self, learners, K=5, loss='L2', optimization_method='slsqp', stop_criteria=0.00001):
        self.K = K
        self.loss = loss
        self.optimization_method = optimization_method
        self.learners = learners # Dictionary of learner name and the corresponding object
        self.n_learners = len(learners.keys())


    def fit(self, X, y):
        n = len(y) # Num data points
        kf = sklearn.model_selection.KFold(n_splits=self.K)
        y_pred = np.empty(shape=(n, self.n_learners))
        # print(y_pred.shape)

        # Iterate throught each split and store the predicted values for each leaner
        for train_index, test_index in kf.split(X):
            # print("TRAIN:", len(train_index), "TEST:", len(test_index))
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            # Iterator through each learner
            for id, value in self.learners.items():
                _, model_object = value
                model = sklearn.clone(model_object)
                model.fit(X_train, y_train)

                if self.loss == 'L2':
                    pred = model.predict(X_test)
                    y_pred[test_index, id] = pred
        
        models = [value[1] for value in self.learners.values()]
        self.trained_model = sklearn.clone(models)
        for model in self.trained_model:
            model.fit(X, y)

        # 1: First Optimizer : Sequential Least SQuares Programming
        if self.optimization_method == 'slsqp':
            # print("Optimizing coefficients using slsqp")
            from scipy.optimize import fmin_slsqp
            def objective(x):
                """Objective function"""
                if self.loss == 'L2':
                    comb = np.dot(y_pred, x)
                    error = np.mean((y - comb)**2)
                return error
            
            def constr(x):
                return np.array([ np.sum(x)-1 ])

            # 1-D ndarray of float; Initial guess for the independent variable(s).
            x0 = np.array([1./self.n_learners] * self.n_learners)
            
            # list; A list of tuples specifying the lower and upper bound for each independent variable [(xl0, xu0),(xl1, xu1),...]
            bounds = [(0,1)] * self.n_learners
            
            # Running the optimizer
            out, fx, its, imode, smode = fmin_slsqp(objective, x0, f_eqcons = constr, bounds = bounds, disp = 0, full_output = 1)
            
            weights = np.array(out)

            # Rounding offf small values to 0
            weights[weights < np.sqrt(np.finfo(np.float64).eps)] = 0

            # Normalized the coefficents withing the range
            weights = weights/np.sum(weights)
            # print(weights)
            self.weights = weights
            if imode != 0:
                raise "Error while calculating coefficients using SLSQP"

        

        self.error_cv = {}
        for id, value in self.learners.items():
            model_name, model_object = value
            
            if self.loss == 'L2':
                error = np.mean((y - y_pred[:,id])**2)
                self.error_cv[model_name] = error 

        if self.loss == 'L2':
            comb = np.dot(y_pred, weights)
            error = np.mean((y - comb)**2)
            self.error_cv['stacked_learner'] = error

    def predict(self , X):
        num_feature = X.shape[0]
        y_pred_mat = np.empty((num_feature, self.n_learners))
        for idx, model in enumerate(self.trained_model):
            if self.loss == 'L2':
                pred = model.predict(X)

            y_pred_mat[:, idx] = pred
        
        if self.loss == 'L2':
            y_pred = np.dot(y_pred_mat, self.weights)

        return y_pred

    @property
    def error(self):
        return self.error_cv['stacked_learner']

    @property
    def coefficents(self):
        return self.weights


if __name__ == "__main__":
    from sklearn import datasets, linear_model, neighbors
    
    # Load Example Dataset for regression
    np.random.seed(100)
    X, y = datasets.make_friedman1(1000)
    
    # All the learners
    leaners = {
        0: ('OLS', linear_model.LinearRegression()),
        1: ('ElasticNetCV', linear_model.ElasticNetCV()),
        2: ('Ridge', linear_model.RidgeCV()),
        3: ('LARS', linear_model.LarsCV()),
        4: ('LASSO', linear_model.LassoCV()),
        5: ('kNN', neighbors.KNeighborsRegressor())
    }
    
    stacked_model = SuperSklearn(leaners)
    stacked_model.fit(X, y)
    y_pred = stacked_model.predict(X)

    print(stacked_model.error)
    print(stacked_model.coefficents)