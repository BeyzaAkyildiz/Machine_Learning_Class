import np as np
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

x_l = np.load('X.npy')
y_l = np.load('Y.npy')
print(x_l.shape)
print(y_l.shape)
img_size = 64
# fotoğrafları ekrana yazdırdık
plt.subplot(1, 2, 1)
plt.imshow(x_l[260].reshape(img_size, img_size))
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(x_l[900].reshape(img_size, img_size))
plt.axis('off')
plt.show()
X = np.concatenate((x_l[204:409], x_l[822:1027] ), axis=0)
# labelları kullanabilmek için uygun formata getirdik
z = np.zeros(205)
o = np.ones(205)
Y = np.concatenate((z, o), axis=0).reshape(X.shape[0],1)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)
#fotoğraflar 64x64 matrixken fotoğrafken 1x4096 haline getirdik
number_of_train = X_train.shape[0]
number_of_test = X_test.shape[0]
X_train_flatten = X_train.reshape(number_of_train,X_train.shape[1]*X_train.shape[2])
X_test_flatten = X_test .reshape(number_of_test,X_test.shape[1]*X_test.shape[2])
#transposunu aldık
x_train = X_train_flatten.T
x_test = X_test_flatten.T
y_train = Y_train.T
y_test = Y_test.T




#one vs all kullanarak model oluşturduk
model = OneVsRestClassifier(LogisticRegression(max_iter=9999))
print(X_train_flatten.shape)
print(Y_train.shape)
#modeli fit ettik
model.fit(X_train_flatten, Y_train)
#prediction yaptık
yhat = model.predict(X_test_flatten)

print("Accuracy:",metrics.accuracy_score(Y_test, yhat)) # bu ve diğer iki satırda tutarlılık isabet skorunu hesapladık
print("Precision:",metrics.precision_score(Y_test, yhat))
print("Recall:",metrics.recall_score(Y_test, yhat))
