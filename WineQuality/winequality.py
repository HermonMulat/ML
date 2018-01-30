import matplotlib.pyplot as plt
from sklearn import linear_model
from scipy.stats import uniform as sp_rand
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import math,sys

def main():
    # Read data
    data = pd.read_csv(sys.argv[1])
    # shuffle
    data.sample(frac=1)

    y = data.quality                    # target
    X = data.drop('quality', axis=1)    # features

    # stratifyed sample spliting
    X_train, X_test, y_train, y_test = train_test_split(X,y,
                                        test_size=0.2, stratify = y)
    model = linear_model.SGDRegressor(max_iter=10)

    # create standardizer
    scaler = StandardScaler()
    scaler.fit(X_train)
    scaled_train_X =  scaler.transform(X_train)
    scaled_test_X =  scaler.transform(X_test)

    # # clf = GridSearchCV(model, {"alpha":[0.00001,0.0001,0.001,0.01,0.1,1],
    # #                             "eta0":[0.00001,0.0001,0.001,0.01,0.1,1]})
    model.fit(scaled_train_X,y_train)
    print model.score(scaled_test_X,y_test)

if __name__ == '__main__':
    main()
