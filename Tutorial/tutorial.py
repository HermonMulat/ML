import random
import matplotlib.pyplot as plt
from sklearn import linear_model

def f(x,m=2.71828, error = 0.1):
    # return a x,y and y = m*x with some error pecentage - default is with 25%
    y = m*x
    e = y*random.uniform(-1*error, error)
    return (x,y+e)

def main():
    values = [f(random.uniform(1,300)) for i in xrange(100)]
    train_set = values[:-20]
    test_set = values[-20:]

    # create LinearRegression model
    model = linear_model.LinearRegression(fit_intercept=False)
    x,y = zip(*train_set)
    x = map(lambda a:[a],x)
    model.fit(x,y)
    m,b = model.coef_[0], model.intercept_
    print "Best fit line: y = {0:.4f}x + {1:.4f}".format(m,b)
    test_x,test_y =  zip(*test_set)
    test_x = map(lambda a:[a], test_x)
    predicted = model.predict(test_x)

    # plot results
    x,y = zip(*train_set)
    plt.scatter(x, y, color='blue') # plot training set
    plt.scatter(map(lambda a:a[0],test_x), test_y, color = "black") # plot test set
    plt.scatter(map(lambda a:a[0],test_x), predicted,color = "red") # plot predicted
    all_x, all_y = zip(*values)
    min_x, max_x = min(all_x),max(all_y)
    y_min, y_max = m*min_x+b, m*max_x + b
    plt.plot([min_x, max_x], [y_min, y_max], color="green") # plot fitted line
    plt.show()

if __name__ == '__main__':
    main()
