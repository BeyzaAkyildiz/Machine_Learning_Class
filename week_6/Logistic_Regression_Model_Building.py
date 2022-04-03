import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
# load dataset
pima = pd.read_csv("diabetes.csv", skiprows=1 , names=col_names) #header=none da headırın eski kolon isimlerinin
# gelmemesi gerekiyordu ama geldiği için hata verdi ve skiprows=1 ile  kolon isimlerini geçtik hata düzeldi

print (pima.head())

#split dataset in features and target variable
feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
X = pima[feature_cols] # Features
y = pima.label # Target variable

# split X and y into training and testing sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
#burada test size için 0.25 ayırdı random state ile birlikte her seferinde bölüm sayısını aynı tutmak
# önemliki farklı olmasın farklı olursa karşılaştırmak zorlaşır burayı isteğe göre biz seçiyoruz

# import the class


# instantiate the model (using the default parameters)
logreg = LogisticRegression(max_iter=999999)#lgistik regrasyonda işleyebileceği bir default limiti var default limit yerine
# kendimiz büyük bir limit belirleyerek hatayı kaldırdık

# fit the model with data
logreg.fit(X_train,y_train)

y_pred=logreg.predict(X_test)

# import the metrics class

cnf_matrix = metrics.confusion_matrix(y_test, y_pred) #confusuion matrix oluşturuldu
print(cnf_matrix)


#matplotlib inline

class_names=['t','f'] # name  of classes #list tanımladı burdaki 0 ve 1 yazısı yerine t ve f
# yazarak nerde olduğunu görmüş olduk ayrıca 55 ve 56. satır buradaydı yeri yanlış olduğu için aşağı koyduk
fig, ax = plt.subplots() #grafik oluşturmada kullanıldı
tick_marks = np.arange(len(class_names)) #  yukardaki dizideki  uzunluk kadar sıralı bir dizi oluşturup
# yukardaki gibi 2 ise uzunluğu 0 1 elemanlarıdır mesela uzunluğu 3 olsaydı 0 1 2 gibi olurdu

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g') #heatmap oluşturduk
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.xticks(tick_marks, class_names) # t ve f leri tick olarak aldlandırıldı
plt.yticks(tick_marks, class_names) # tick_marks tiklerin sayısını belirledi class names her tick in adını belirledik
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()
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

