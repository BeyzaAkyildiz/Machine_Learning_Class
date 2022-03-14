import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.model_selection import train_test_split

knn = KNeighborsClassifier(n_neighbors=3)


def load_2C_data():
    csv_path = "input/2C.csv"
    return pd.read_csv(csv_path)


data = load_2C_data()  # dataya dataseti attı

x, y = data.loc[:, data.columns != 'class'], data.loc[:, 'class']
knn.fit(x, y)
prediction = knn.predict(x)
print('Prediction: {}'.format(prediction))

#dataseti label ve future olarak ayırdık , fit ettik sonra prediction yaptık knn ile birlikte

# train test split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 1)
knn = KNeighborsClassifier(n_neighbors = 3) # k yi 3 verdik
x,y = data.loc[:,data.columns != 'class'], data.loc[:,'class']
# seçim yaptık class olup olmamasına göre
knn.fit(x_train,y_train)  #eğittik
prediction = knn.predict(x_test) #test ettik
#print('Prediction: {}'.format(prediction))
print('With KNN (K=3) accuracy is: ',knn.score(x_test,y_test))

neig = np.arange(1, 25)
train_accuracy = []
test_accuracy = []
# Loop over different values of k
for i, c in enumerate(neig):
    # k from 1 to 25(exclude)
    knn = KNeighborsClassifier(n_neighbors=c)
    # Fit with knn
    knn.fit(x_train,y_train)
    #train accuracy
    train_accuracy.append(knn.score(x_train, y_train))
    # test accuracy
    test_accuracy.append(knn.score(x_test, y_test))

# Plot
plt.figure(figsize=[13,8])
plt.plot(neig, test_accuracy, label = 'Testing Accuracy')
plt.plot(neig, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.title('-value VS Accuracy')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.xticks(neig)
plt.savefig('graph.png')
plt.show()
print("Best accuracy is {} with K = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))

#modeli eğittik ama eni iyi k yı bulmak için bu denemeleri yaptık

