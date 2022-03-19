import pandas as pd
import seaborn
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv('Admission_Predict_Ver1.1.csv')
data.head()

f,ax = plt.subplots(figsize=(18, 18))
seaborn.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
# burada bir çok future var ve korelasyon oranına baktım future ve label seçtim
# CGPA i future olarak seçtim ve label olarakda   Chance of Admit i seçtim burda 1 vve 0 var
# koralasyon miktarı bu ikisinin 0.9 yani çok iyi

data.plot(kind='scatter', x='CGPA', y='Chance_of_Admit',alpha = 0.5,color = 'red')
plt.xlabel('CGPA')              # label = name of label
plt.ylabel('Chance_of_Admit')
plt.title('Impact of CGPA grade on Chance of Admit')
plt.show() #bu yukardaki gibi özellikleri verilen tabloyu ekrana getirdi/gösterdi
# veri scatter formatında görüntüledim


# linear regression model
linear_reg = LinearRegression()  # modeli değişkene atadı

x = data.CGPA.values.reshape(-1,1)
y = data.Chance_of_Admit.values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 123, shuffle=1)
#burda test ve train verimizi böldük

linear_reg.fit(x,y)

# prediction

prediction = linear_reg.predict(X_test)

plt.plot(X_test, prediction,color = "red")
plt.scatter(x,y)
plt.show()
# burada prediction çizgimizi yazdırdık

print('R^2 score: ',linear_reg.score(x, y)) #böylece r2 score umu bulmuş oldum



