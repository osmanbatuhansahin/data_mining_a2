import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.datasets import make_classification
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import mean_squared_error
import time
import warnings
warnings.filterwarnings("ignore")
from sklearn.svm import SVC
from sklearn import svm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neural_network import MLPClassifier


#Task 2.1

print("Task 2.1")
def task2_1(tuple,dimension):
    X, y = make_classification(n_samples=tuple, n_features=dimension)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.7)
    perc = SGDClassifier(loss="perceptron", max_iter=iterations)

    perc.fit(X_train,y_train)
    y_predict = perc.predict(X_test)
    error = mean_squared_error(y_true=y_test,y_pred=y_predict)
    return error

time_list = []
error_list = []

for i in [10000,100000,250000]:
    for j in range (10):
        iterations = 100
        start = time.time()
        error_list.append(task2_1(i,100))
        stop = time.time()
        time_list.append(stop-start)

    print("iterations:100","tuple:",i,"dimension=100","-> Error:",sum(error_list) / 10,"Time:",(sum(time_list) / 10)*1000,"ms")
    time_list = []
    error_list = []

for i in [100000]:
    for j in range (10):
        iterations = 100
        start = time.time()
        error_list.append(task2_1(i,1000))
        stop = time.time()
        time_list.append(stop-start)
    print("iterations:100","tuple:",i,"dimension=1000","-> Error:",sum(error_list) / 10,"Time:",(sum(time_list) / 10)*1000,"ms")
    time_list = []
    error_list = []

for i in [10000,100000,250000]:
    for j in range (10):
        iterations = 500
        start = time.time()
        error_list.append(task2_1(i,100))
        stop = time.time()
        time_list.append(stop-start)
    print("iterations:500","tuple:",i,"dimension=100","-> Error:",sum(error_list) / 10,"Time:",(sum(time_list) / 10)*1000,"ms")
    time_list = []
    error_list = []

for i in [100000]:
    for j in range (10):
        iterations = 500
        start = time.time()
        error_list.append(task2_1(i,1000))
        stop = time.time()
        time_list.append(stop-start)
    print("iterations:500","tuple:",i,"dimension=1000","-> Error:",sum(error_list) / 10,"Time:",(sum(time_list) / 10)*1000,"ms")
    time_list = []
    error_list = []

#Task 2.2

print("Task 2.2")

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], random_state=0, train_size=0.7)
A = X_test[:, :3]
B = y_test


A = A[np.logical_or(B==0,B==1)]
B = B[np.logical_or(B==0,B==1)]

perc = SGDClassifier(loss="perceptron", max_iter=100)
model = svm.SVC(kernel='linear')
clf = perc.fit(A, B)
y_predict = perc.predict(A)



z = lambda x,y: (-clf.intercept_[0]-clf.coef_[0][0]*x -clf.coef_[0][1]*y) / clf.coef_[0][2]

tmp = np.linspace(-5,5,30)
x,y = np.meshgrid(tmp,tmp)

fig = plt.figure()
ax  = fig.add_subplot(111, projection='3d')
ax.plot3D(A[B==0,0], A[B==0,1], A[B==0,2],'ob')
ax.plot3D(A[y_predict==1,0], A[y_predict==1,1], A[y_predict==1,2],'sr')
ax.plot_surface(x, y, z(x,y))
ax.view_init(30, 60)
plt.show()


#Task 3.1

print("Task 3.1")

digits = load_digits()

X_train, X_test, y_train, y_test = train_test_split(digits['data'],digits['target'], random_state=0, test_size=0.3, train_size=0.7)
mlper = MLPClassifier(hidden_layer_sizes=(50,), max_iter=100)
mlper.fit(X_train, y_train)
prediction = mlper.predict(X_test)
error = mlper.loss_curve_
plt.title('Convergence of error with MLP')
plt.xlabel("iteration")
plt.ylabel("error")
plt.plot(error,color = "red")
plt.show()


#Task 3.2

print("Task 3.2")
def task3_2(h):
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(digits['data'],digits['target'], test_size=0.3, train_size=0.7, random_state=42)
    mlper = MLPClassifier(hidden_layer_sizes=(h), max_iter=100, random_state=42)
    mlper.fit(X_train, y_train)

    score = mlper.score(X_train, y_train)
    score2 = mlper.score(X_test, y_test)
    return score, score2


scoretrain_list = []
scoretest_list = []

scoretrain_list.append(task3_2(2)[0])
scoretrain_list.append(task3_2((4,2))[0])
scoretrain_list.append(task3_2((8,4,2))[0])
scoretrain_list.append(task3_2((16,8,4,2))[0])
scoretrain_list.append(task3_2((32,16,8,4,2))[0])
scoretrain_list.append(task3_2((64,32,16,8,4,2))[0])
scoretrain_list.append(task3_2((128,64,32,16,8,4,2))[0])
scoretrain_list.append(task3_2((256,128,64,32,16,8,4,2))[0])
scoretrain_list.append(task3_2((512,256,128,64,32,16,8,4,2))[0])
scoretrain_list.append(task3_2((1024,512,256,128,64,32,16,8,4,2))[0])

scoretest_list.append(task3_2(2)[1])
scoretest_list.append(task3_2((4,2))[1])
scoretest_list.append(task3_2((8,4,2))[1])
scoretest_list.append(task3_2((16,8,4,2))[1])
scoretest_list.append(task3_2((32,16,8,4,2))[1])
scoretest_list.append(task3_2((64,32,16,8,4,2))[1])
scoretest_list.append(task3_2((128,64,32,16,8,4,2))[1])
scoretest_list.append(task3_2((256,128,64,32,16,8,4,2))[1])
scoretest_list.append(task3_2((512,256,128,64,32,16,8,4,2))[1])
scoretest_list.append(task3_2((1024,512,256,128,64,32,16,8,4,2))[1])



x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
plt.plot(x, scoretrain_list, 'b', label='Train', marker="+")
plt.plot(x, scoretest_list, 'r', label='Test', marker="o")
plt.legend(loc='best')
plt.xlabel("Hidden Layer Size")
plt.ylabel("Accuracy Score")
plt.show()











