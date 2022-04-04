import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv("framingham.csv")
print (data.head())

imputer = SimpleImputer(strategy="median")  # datasetteki boş verileri ortalaması nı alıp bu ortalamaları
# boş değerlere atar

data2 = imputer.fit_transform(data)  # data içindeki tüm boş değerleri empoze eder

data = pd.DataFrame(data2, columns=data.columns, index=data.index)  # pandas dataframe i oluşturuduk


feature_cols = ['male','age','education','currentSmoker','cigsPerDay','BPMeds','prevalentStroke','prevalentHyp',
                'diabetes','totChol','sysBP','diaBP','BMI','heartRate','glucose']

x = data[feature_cols] # Features
y = data['TenYearCHD'] # Target variable

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

logreg = LogisticRegression(max_iter=999999)

logreg.fit(X_train,y_train)

y_pred=logreg.predict(X_test)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred) #confusuion matrix oluşturuldu
print(cnf_matrix)

class_names=[1,0] # name  of classes #list tanımladık
fig, ax = plt.subplots() #grafik oluşturmada kullanıldı
tick_marks = np.arange(len(class_names))

print(logreg.score(X_test,y_test))

print("Accuracy:",metrics.accuracy_score(y_test, y_pred)) # bu ve diğer iki satırda tutarlılık isabet skorunu hesapladık
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))

y_pred_proba = logreg.predict_proba(X_test)[::,1]  #Sınıflandırma problemlerinde gözlemlerin sınıflara
# ait olma olasılıklarını elde etmek istiyorsak predict_proba fonksiyonunu kullanılır bu yüzden kullandık
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
