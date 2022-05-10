import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

data = pd.read_csv('data.csv')

data.drop(["id","Unnamed: 32"],axis=1,inplace=True)  #2 kolon sildik
# malignant = M  kotu huylu tumor
# benign = B     iyi huylu tumor
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis] #m se 1 değilse 0 olarak değiştirdik
y = data.diagnosis.values #yeni label diziyi y ye attı
x = data.drop(["diagnosis"],axis=1) #label ı sildik böylece sadece futurelar kaldı
# %%
# normalization  ortalama yöntemiyle ayarlıyor burada hata veriyordu x_datanın
# .min() ve .max() kısımları sonuna eklediğimizde çözüldü
x = (x - x.min())/(x.max()-x.min())

# train test split train ve test verisini ayırdık
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state=1)

svm = SVC(C=1)
# Fit with svm
svm.fit(x_train, y_train)
# train accuracy
s_train_accuracy=(svm.score(x_train, y_train))
# test accuracy
s_test_accuracy=(svm.score(x_test, y_test))

navie=GaussianNB()
navie.fit(x_train, y_train)
n_train_accuracy=(navie.score(x_train, y_train))
n_test_accuracy=(navie.score(x_test, y_test))

print("accuracy of svm: ",s_test_accuracy)

print("accuracy of navie: ",n_test_accuracy)

