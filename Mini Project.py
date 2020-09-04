import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
df_train = pd.read_csv('mnist_train.csv')
df_test = pd.read_csv('mnist_test.csv')
##print(df_train.head(5))
dfarray_train = df_train.to_numpy()
X_train1 = dfarray_train[:,1:]
Y_train1 = dfarray_train[:,0]
#plt.imshow(X_train[10].reshape(28,28), cmap='gray')
#plt.show()
##print(Y_train)
dfarray_train_prm = dfarray_train[np.random.RandomState(seed=42).permutation(dfarray_train.shape[0])]
X_train = dfarray_train_prm[:,1:]
Y_train = dfarray_train_prm[:,0]
dfarray_train_prm.shape
#plt.imshow(X_train[4].reshape(28,28), cmap='gray')
#plt.show()
X_train = X_train/255
X_train2= X_train[ :4800,:]
Y_train2 = Y_train[: 4800]

X_train_validation = X_train[4800:6000,: ]
Y_train_validation = Y_train[4800:6000]
dfarray_test = df_test.to_numpy()
X_test = dfarray_test[:,1:]
Y_test = dfarray_test[:,0]
X_test = X_test/255
"""logistic regression"""
clf = LogisticRegression(random_state=42, max_iter=700, verbose=1, multi_class='ovr', n_jobs=-1)
clf.fit(X_train2, Y_train2)
print('the score for logisticregression is: '+ str(clf.score(X_train2, Y_train2)))
"""knn"""
j=0
a=0
i=1
for i in range(2,10):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train2, Y_train2)
    k=knn.score(X_train2, Y_train2)
    print('now '+str(i))
    if k>j:
        j=k
        a=i
print('the best score for train is:'+str(j)+' for '+str(a)+' nearest neighbours')
j=0
a=0
i=1
h=0
for i in range(2,10):
    knn2 = KNeighborsClassifier(n_neighbors=i,n_jobs=-1)
    knn2.fit(X_train2, Y_train2)
    k=knn2.score(X_train_validation, Y_train_validation)
    print('v_now '+str(i))
    if k>j:
        knn_test = KNeighborsClassifier(n_neighbors=i,n_jobs=-1)
        knn_test.fit(X_train2, Y_train2)
        z=knn_test.score(X_test,Y_test)
        j=z
        h=k
        a=i
print('the best score for validation is:'+str(h)+' for '+str(a)+' nearest neighbours'+' and the score for test is '+str(j))

"""svm"""
a=''
b=0
c=0
h=0
svm = SVC(C=1.0, kernel='linear', probability=True)
svm.fit(X_train2, Y_train2)
kernel_set =['linear', 'poly', 'rbf', 'sigmoid']
for k in kernel_set:
    print(k)
    svm = SVC(C=1.0, kernel=k, probability=True)
    svm.fit(X_train2, Y_train2)
    b=svm.score(X_train_validation, Y_train_validation)
    if b>c:
        c=b
        a=k
        svm_test = SVC(C=1.0, kernel=k, probability=True)
        svm.fit(X_train2, Y_train2)
        h=svm.score(X_test,Y_test)
print('the best score is for '+a+' and is : '+str(c))
print('\nand the score for test is '+str(h))
"""neural network"""
NN = MLPClassifier(hidden_layer_sizes=(50, 50, 50,), max_iter=1000)
NN.fit(X_train2, Y_train2)
print(NN.score(X_train2, Y_train2))
score_train_NN = []
hidden_layer_size_set =[]
for k in range(5,50,5):
    NN = MLPClassifier(hidden_layer_sizes=k, max_iter=1000) 
    NN.fit(X_train2, Y_train2)
    score_train_NN.append(NN.score(X_train_validation, Y_train_validation))
    hidden_layer_size_set.append(k)
print('Best Layer Size = ', hidden_layer_size_set[np.argmax(score_train_NN)])
print('Best Score for NN = ', np.max(score_train_NN))
NN_opt_layer_size = MLPClassifier(hidden_layer_sizes=45, max_iter=1000) 
NN_opt_layer_size.fit(X_train2, Y_train2)
print('score for test sample '+str(NN_opt_layer_size.score(X_test,Y_test)))
"""decision tree"""
score_train_dt = []
num_trees =[]
num_max_sample = []
for k in range(20,100,10):
    for s in range(50,300,50):
        clf = BaggingClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=k,  max_samples=s, n_jobs=-1)
        clf.fit(X_train2, Y_train2)
        score_train_dt.append(clf.score(X_train_validation, Y_train_validation))
        num_trees.append(k)
        num_max_sample.append(s)
print('Best number of trees : '+ str(num_trees[np.argmax(score_train_dt)]))
print('and Best number of max_sample : '+str(num_max_sample[np.argmax(score_train_dt)]))
print('Optimum Score for DecisionTree = '+ str(np.max(score_train_dt)))
