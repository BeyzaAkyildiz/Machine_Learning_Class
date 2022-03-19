import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("multiple-linear-regression-dataset.csv",sep = ";")


x = df.iloc[:,[0,2]].values # tüm satırları aldık 0 ve 2. kolonları aldık
y = df.salary.values.reshape(-1,1) #sutun haline getirdi dikey

# %% fitting data
# #modeli tanımladık ve eğittik
multiple_linear_regression = LinearRegression()
multiple_linear_regression.fit(x,y)

print("b0: ", multiple_linear_regression.intercept_)
print("b1,b2: ",multiple_linear_regression.coef_)

# predict
print(multiple_linear_regression.predict(np.array([[10,35],[5,35]])))


df.plot()
plt.show()
#böylece çoklu future u olan değişkenli yaptık