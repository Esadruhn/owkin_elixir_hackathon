import re

import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression

import numpy as np

import substratools as tools


class Algo(tools.algo.AggregateAlgo):
 

    def aggregate(self, models, rank):
        
        coefs = [model.coef_ for model in models]
        intercepts = [model.intercept_ for model in models]

        clf = LogisticRegression()
        clf.coef_ = np.mean(coefs, axis=0)
        clf.intercept_ = np.mean(intercepts, axis=0)

        return clf

    def load_model(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def save_model(self, model, path):
        with open(path, 'wb') as f:
            pickle.dump(model, f)


if __name__ == '__main__':
    tools.algo.execute(Algo())
