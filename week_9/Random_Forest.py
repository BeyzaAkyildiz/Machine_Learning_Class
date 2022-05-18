import pandas as pd


# %%  data

data = pd.read_csv("data.csv")
data.drop(["id", "Unnamed: 32"], axis=1, inplace=True)

# 1 ve 0 olarak sınıflandırıldı
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
y = data.diagnosis.values
x_data = data.drop(["diagnosis"], axis=1)
# %% normalizasyon

x = (x_data - x_data.min()) / (x_data.max() - x_data.min())

# %% train verisi ayrıldı
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)

# %% decision tree
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
print("decision tree score: ", dt.score(x_test, y_test))

# %%  random forest
from sklearn.ensemble import RandomForestClassifier

for i in range(1, 50):
    rf = RandomForestClassifier(n_estimators=i, random_state=1)
    rf.fit(x_train, y_train)
    print("random forest algo result(i={}): ".format(i), rf.score(x_test, y_test))


