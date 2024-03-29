import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('data.csv')

data.drop(["id","Unnamed: 32"],axis=1,inplace=True)  #2 kolon sildik
# malignant = M  kotu huylu tumor
# benign = B     iyi huylu tumor

data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis] #m se 1 değilse 0 olarak değiştirdik
y = data.diagnosis.values #yeni label diziyi y ye attı
x = data.drop(["diagnosis"],axis=1) #label ı sildik böylece sadece futurelar kaldı


x = (x - x.min())/(x.max()-x.min())

# train ve test verisini ayırdık
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state=1)


neig = np.arange(1, 100)
train_accuracy = []
test_accuracy = []

# c lere göre tek tek model oluşturup test ettik
for i, c in enumerate(neig):
    # c from 1 to 100(exclude)
    dt = DecisionTreeClassifier(max_depth=c,random_state=0)
    # Fit with svm
    dt.fit(x_train,y_train)
    #train accuracy #append train_accuracy = [],test_accuracy = [] bu kisine eleman eklemeye yarar
    train_accuracy.append(dt.score(x_train, y_train))
    # test accuracy
    test_accuracy.append(dt.score(x_test, y_test))

# Plot
plt.plot(range(1, 100), test_accuracy)
plt.xlabel("max_depth values")
plt.ylabel("accuracy")
plt.show()

print("Best accuracy is {} with max_depth values = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))

