from sklearn.preprocessing import StandardScaler, MinMaxScaler
from itertools import combinations
import pandas as pd
import sys

def addRatioterms(data):
    headers = data.columns.difference(['quality'])
    # scale columns so we have no zero terms
    minMaxer = MinMaxScaler(feature_range = (1,2))
    data[headers] = minMaxer.fit_transform(data[headers])
    for col1,col2 in combinations(headers,2):
        data[str(col1)+"/"+str(col2)] = data.apply(lambda x: x[col1]/x[col2], axis = 1)
        data[str(col2)+"/"+str(col1)] = data.apply(lambda x: x[col2]/x[col1], axis = 1)

    return data

def addQuadterms(data):
    headers = data.columns.difference(['quality'])
    # adding mixed terms
    for col1,col2 in combinations(headers,2):
        data[str(col1)+"*"+str(col2)] = data.apply(lambda x: x[col1]*x[col2], axis = 1)

    #adding single-squared terms
    for col in headers:
        data[str(col)+"**2"] = data.apply(lambda x: x[col]**2, axis = 1)

    return data

def addRatioQuadterms(data):
    headers = data.columns.difference(['quality'])
    minMaxer = MinMaxScaler(feature_range = (1,2))
    data[headers] = minMaxer.fit_transform(data[headers])
    # adding mixed terms
    for col1,col2 in combinations(headers,2):
        data[str(col1)+"*"+str(col2)] = data.apply(lambda x: x[col1]*x[col2], axis = 1)

    #adding single-squared terms
    for col in headers:
        data[str(col)+"**2"] = data.apply(lambda x: x[col]**2, axis = 1)

    for col1,col2 in combinations(headers,2):
        data[str(col1)+"/"+str(col2)] = data.apply(lambda x: x[col1]/x[col2], axis = 1)
        data[str(col2)+"/"+str(col1)] = data.apply(lambda x: x[col2]/x[col1], axis = 1)

    return data

def main():

    data = pd.read_csv(sys.argv[1], sep = ";")
    # Min max scale
    if sys.argv[3] == "ratio":
        data = addRatioterms(data)
    elif sys.argv[3] == "quad":
        data = addQuadterms(data)
    elif sys.argv[3] == "both":
        data = addRatioQuadterms(data)
    data.to_csv(sys.argv[2])

if __name__ == '__main__':
    main()
