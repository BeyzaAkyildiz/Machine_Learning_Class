import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv('Admission_Predict_Ver1.1.csv')

x = data.iloc[:,[1,2,6]].values # tüm satırları aldık 0 ve 2. kolonları aldık
y = data.Chance_of_Admit.values.reshape(-1,1) #sutun haline getirdi dikey


# %% fitting data
# #modeli tanımladık ve eğittik
multiple_linear_regression = LinearRegression()
multiple_linear_regression.fit(x,y)

print("b0: ", multiple_linear_regression.intercept_) #futureumuz 0 ken y değeri kaç olduğu
print("b1,b2: ",multiple_linear_regression.coef_) # burada ağırlık ve future değeri

# predict
print(multiple_linear_regression.predict(np.array([[311,112,8.3],[319,119,9.5]])))

#burada test verisini elle verdik  3 future um olduğu için bir testte 3 değer verildi



