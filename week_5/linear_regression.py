import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv("linear-regression-dataset.csv",sep = ";") #; ile ayırmış dateseti

plt.scatter(df.experience,df.salary)
#Scatter Plot (Serpilme Diyagramı) iki farklı değer arasındaki ilişkiyi belirlemek için kullanılan ve noktalardan oluşan bir tablodur
plt.xlabel("experience")
plt.ylabel("salary")
plt.show()

#%% linear regression
# sklearn library
from sklearn.linear_model import LinearRegression


# linear regression model
linear_reg = LinearRegression()  # modeli değişkene atadı

x = df.experience.values.reshape(14,1)
y = df.salary.values.reshape(14,1)
#dizi boyutu gibi satır ve sutunu temsil ediyor stunun hepsini getiriyor 14 yerine -1 de yazabiliriz

linear_reg.fit(x,y)

#%% prediction


b0 = linear_reg.predict([[0]]) # tahmin ediyor ve bu tahminle veri setini karşılaştırıcak ??
print("b0: ",b0)

b0_ = linear_reg.intercept_ #bias y ekseniyle kesişimi
print("b0_: ",b0_)   # intercept the y-axis

b1 = linear_reg.coef_ #eğimi hesapladık çizgimizin tahminimizin
print("b1: ",b1)   # slope

# salary = 1663 + 1138*experience

new_salary = 1663 + 1138*11 #prediction çizgisinin y sini bulmaya yarar
print(new_salary)

b11 = linear_reg.predict([[11]]) # 11 i tahmin etmeye çalışıyo futureı vermiş label ı  istiyor
print("b11: ",b11)

# visualize line
array = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).reshape(-1,1)  # experience

# yeni bir array oluşturduk bunu dikine sıraladık bissürü satır tek bir kolon haline getrirdi

plt.scatter(x,y) #scatter grafiği oluşturduk
plt.show()

y_head = linear_reg.predict(array)  # salary
# yukarda tanımladığımız diziyi tahmin ettirdik
plt.plot(array, y_head,color = "red")

b100 = linear_reg.predict([[100]])
print("b100: ",b100)
# 100 ün tahminin değerini yazdırık
plt.show()



