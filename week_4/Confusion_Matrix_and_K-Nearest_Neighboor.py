# KNN Algorithm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# %%
data = pd.read_csv("input/data.csv")

# %%
# data.drop kaldırıyr siler aşağıdaki belirtilen yerdeki verileri
data.drop(["id","Unnamed: 32"],axis=1,inplace=True) # burada id kısmını ve Unnamed: 32 kısmmını 1 dediği için dikey olarak siler inplace
# kısmını false yaparsak return eder ve veri değiştirilmeden bir değişkene atılırsa korunmuş olur
print(data.tail().to_string())# burda printi ben yazarak yazdırdım .to_string() yaparak ise tablodaki tüm verileri göstermesini sağladım


# malignant = M  kotu huylu tumor
# benign = B     iyi huylu tumor


# %%
M = data[data.diagnosis == "M"]
B = data[data.diagnosis == "B"]  # veriyi ikiye böldü ayırdı
# scatter plot ---> bu bir tablo türüdür
plt.scatter(M.radius_mean,M.texture_mean,color="red",label="kotu",alpha= 0.3) # m leri kırmızı baloncuklarla kötü olrak tabloya ekledi
plt.scatter(B.radius_mean,B.texture_mean,color="green",label="iyi",alpha= 0.3)# b leri yeşil baloncuklarla iyi olarak tabloya ekledi
plt.xlabel("radius_mean") #yazdırdı
plt.ylabel("texture_mean") #yazdırdı
plt.legend() # iyi kötü yazan konumu ve özelliklerini içinde yzıyor ama gözükmüyor
plt.show()#ekrana getirdi yazdırdı

# %%
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
y = data.diagnosis.values
x_data = data.drop(["diagnosis"],axis=1)
#m olanları bir diğerleri sıfır yapar m kötü huylu tümör
# %%
# normalization
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))
#burada Feature Scaling yapılmıştır bunun sonucunda  değişkenlerin farklı ortalamalarda ve standart sapmaya sahip olmasına
# izin verilir ancak aynı aralıkta olmaları şartıyla
#%%
# train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state=1)
# train ve test  set i ayarladı
# %%
# knn model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3) # n_neighbors = k
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
print(" {} nn score: {} ".format(3,knn.score(x_test,y_test)))
#modeli oluşturdu
# %%
# find k value
score_list = []
for each in range(1, 15):
    knn2 = KNeighborsClassifier(n_neighbors=each)
    knn2.fit(x_train, y_train)
    score_list.append(knn2.score(x_test, y_test))

plt.plot(range(1, 15), score_list)
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.show()

y_pred = knn2.predict(x_test)
y_true = y_test

#en iyi k değerini bulmaya çalıştı
#%% confusion matrix oluşturulması
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true,y_pred)


# %% cm visualization
import seaborn as sns
import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()

# grafiğin ayarlamasını yaptı