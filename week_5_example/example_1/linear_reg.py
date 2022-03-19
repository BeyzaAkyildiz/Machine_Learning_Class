# create data1 that includes pelvic_incidence that is feature and sacral_slope that is target variable
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

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

# LinearRegression
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
# Predict space
predict_space = np.linspace(min(x), max(x)).reshape(-1,1) # min ile max arasında defoult olarak bölme sayısı 50 dir
# ve adım sayıları her elii küme için aynıdır
# Fit
reg.fit(x,y)
# Predict
predicted = reg.predict(predict_space)
# R^2
print('R^2 score: ',reg.score(x, y))
# Plot regression line and scatter
plt.plot(predict_space, predicted, color='black', linewidth=3)
plt.scatter(x=x,y=y)
plt.xlabel('pelvic_incidence')
plt.ylabel('sacral_slope')
plt.show()