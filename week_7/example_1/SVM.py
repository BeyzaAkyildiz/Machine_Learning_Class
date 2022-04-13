import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix


data = pd.read_csv("data.csv")
print (data.head())
data.drop(["id","Unnamed: 32"],axis=1,inplace=True)  #2 kolon sildik
# malignant = M  kotu huylu tumor
# benign = B     iyi huylu tumor

# %% tablonun hazırlanışı
M = data[data.diagnosis == "M"]
B = data[data.diagnosis == "B"]
# scatter plot
plt.scatter(M.radius_mean,M.texture_mean,color="red",label="kotu",alpha= 0.3)
plt.scatter(B.radius_mean,B.texture_mean,color="green",label="iyi",alpha= 0.3)
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.legend() #ufak simge kısmı tanımlamak için sınıflandırma yaptığımız durumları gösterdik
plt.show()

# %%
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis] #m se 1 değilse 0 olarak değiştirdik
y = data.diagnosis.values #yeni label diziyi y ye attı
x_data = data.drop(["diagnosis"],axis=1) #label ı sildik böylece sadece futurelar kaldı



# %%
# normalization  ortalama yöntemiyle ayarlıyor burada hata veriyordu x_datanın
# .min() ve .max() kısımları sonuna eklediğimizde çözüldü
x = (x_data - x_data.min())/(x_data.max()-x_data.min())

#%%
# train test split train ve test verisini ayırdık
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state=1)

# %% SVM
svm = SVC(random_state=1) #modeli tanımladı
svm.fit(x_train, y_train) #modeli eğitti


# %% test
print("print accuracy of svm algo: ",svm.score(x_test,y_test)) #modelin skorunu hesapladık

y_pred = svm.predict(x_test) #burda test ettik modeli
print(y_pred)

# Confusion Matrix
print(confusion_matrix(y_test, y_pred)) #confusion matrix i hesapladık
