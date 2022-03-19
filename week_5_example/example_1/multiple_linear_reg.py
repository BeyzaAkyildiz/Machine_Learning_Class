from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

data = pd.read_csv('2c_5.csv')
data1 = data[data['class'] =='Abnormal']  #sadece abnormal olanların hepsini data1 içine koydu
x = np.array(data1.loc[:,['pelvic_incidence','lumbar_lordosis_angle','degree_spondylolisthesis','pelvic_tilt numeric']]).reshape(-1,4) # futureları belirledik
y = np.array(data1.loc[:,'sacral_slope']).reshape(-1,1) #aynı şeyi sacral içi yaptı

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)


# %% fitting data
# #modeli tanımladık ve eğittik
multiple_linear_regression = LinearRegression()
multiple_linear_regression.fit(X_train,y_train)

print("b0: ", multiple_linear_regression.intercept_) #futureumuz 0 ken y değeri kaç olduğu
print("b1,b2: ",multiple_linear_regression.coef_) # burada ağırlık ve future değeri


#burada test verisini elle verdik  3 future um olduğu için bir testte 3 değer verildi
f,ax = plt.subplots(figsize=(18, 18))#kutucuğun boyutu 2 tan değer var
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
y_head=multiple_linear_regression.predict(X_test)

print("r_square score for multiple linear regression: ", r2_score(y_test,y_head))

