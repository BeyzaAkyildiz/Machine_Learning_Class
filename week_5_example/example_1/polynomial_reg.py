from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv('2c_5.csv')

data1 = data[data['class'] =='Abnormal']  #sadece abnormal olanların hepsini data1 içine koydu
x = np.array(data1.loc[:,'pelvic_incidence']).reshape(-1,1) # atributse  pelvik olan tüm satırı aldı
y = np.array(data1.loc[:,'sacral_slope']).reshape(-1,1) #aynı şeyi secral içi yaptı


# Scatter
plt.figure(figsize=[10,10])
plt.scatter(x=x,y=y)
plt.xlabel('pelvic_incidence')
plt.ylabel('sacral_slope')
plt.show()

#linear regression yapıldı
linear_reg = LinearRegression()  # modeli değişkene atadı

linear_reg.fit(x,y)
#prediction yapıldı
y_head = linear_reg.predict(x)

plt.scatter(x,y)
plt.xlabel("CGPA")
plt.ylabel("Chance_of_Admit")

plt.plot(x,y_head,color="red",label ="linear")
plt.show()

#polinomal reg yapıldı

polynomial_regression = PolynomialFeatures(degree = 2)

x_polynomial = polynomial_regression.fit_transform(x)

#fit edildi

linear_regression2 = LinearRegression()
linear_regression2.fit(x_polynomial,y)
##

y_head2 = linear_regression2.predict(x_polynomial)

plt.scatter(x,y)
plt.xlabel("CGPA")
plt.ylabel("Chance_of_Admit")
plt.plot(x,y_head,color="red",label ="linear")
#bu linear için pred çizgisi

plt.plot(x,y_head2,color= "green",label = "poly")
plt.legend()
plt.show()
#bu polinomal için pred çizgisi

# polinomal pred çizgisi gösterildi yeşil olan polinomal diğeri linear

print("r_square score for linear regression: ", r2_score(y,y_head))

print("r_square score for polynomial regression: ", r2_score(y,y_head2))

#r2 skor olarak polinomal la linear yakın zaten ama polinomal bir tık daha iyi
# aradaki fark 0.004 küsür brişey
