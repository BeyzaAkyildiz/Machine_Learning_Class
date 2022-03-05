

#week_3 odev için week_2 kodlar gerekli olduğu için duruyor ve week_3 ün başlangıcı belirtilmiştir

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from zlib import crc32
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer


def load_housing_data():
    csv_path = "../input/housing.csv"
    return pd.read_csv(csv_path)


housing = load_housing_data()
print(housing.head())  # başları yazdı
print(housing.info())  # atributes özelliklerini indexiyle birlikte yazdırdık
print(housing["ocean_proximity"].value_counts())  # ocean_proximity bir atributes ve altında feature lar  var ve bu ,
# sayılar o feature nin toplam feature sayısını vermiş oldu
print(
    housing.describe().to_string())  # her kolon için count,ortalama(mean),std(standart sapma),min,max,%25 gibi hesaplama sonuçlarını verir

housing.hist(bins=50, figsize=(20, 15))  # histogram yani çubuk grafik hesapladı verileri ve gösterdi
# burdaki tabloda bins olan yer görsellleştirilmiş tablodaki verileri gösteren çubuk sayısıdır çubuk saysısı azalıkça
# verilerde genlelleme gösterir medsela 0-10 -10-20 iken binsi artırırsak 0-5  5-10 10-15 15-20 arasındaki değerleri gösterir
# figsize ise görsel tablo açıkken tüm resmin boyutunu ayarlar
# plot yazdığımız zaman tek bir grafikte hepsini göstermeye çalışıyor atributesları housing.plot(kind='hist', bins=50, figsize=(20,15))
# hist yaptığımız zaman her atributes ayrı grafik gösteriliyor housing.hist(bins=50, figsize=(20,15)) gibi
# birsde her bir tabloyu plot ile ayrı yazmak istiyorsak housing.housing_median_age.plot(kind='hist', bins=50, figsize=(20,15)) mesela böyle yazıyoruz

plt.show()


def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(
        len(data))  # burda datanın uzunluğuna göre data indeklerini karıştırır mesela
    # len 4 olsun  3 1 2 0 diye karmaşık sıralarsa d b c a diye datamız karışmış oldu indekleri tekrar 0 dan başlaması return de oldu
    test_set_size = int(
        len(data) * test_ratio)  # test radio dediğimiz şey test verisine yüzde kaç veri ayıracağı toplam veri setinden
    # böylece trest verisi sayısına ulaştık
    test_indices = shuffled_indices[:test_set_size]  # burda test_set_zize 2 alırsa 0 ve 1 indisleri seçildi

    train_indices = shuffled_indices[test_set_size:]
    # burda ise 2 den soraki dediği için 2 ve 3 indisleri buranın oldu
    return data.iloc[train_indices], data.iloc[
        test_indices]  # burda ise verilen indislere göre test ısmı için a ve b train datası olarakda c ve d
    # datalarını paylaştırdı


train_set, test_set = split_train_test(housing, 0.2)  # 0.2 test veris oranını verdi buna göre hesaplanacak
print("train set: ", len(train_set))

print("test set: ", len(test_set))


def test_set_check(identifier, test_ratio):  # yukardakinin daha ayrıntılı halidir aynı sonuçları verir ?????
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2 ** 32


def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


housing_with_id = housing.reset_index()  # adds an `index` column indexi kolona ekledik çünkü id verebilmek için
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

housing_with_id["id"] = housing["longitude"] * 1000 + housing[
    "latitude"]  # bu yöntemle id leri belirledik evin konumu eşsiz
# olduğu için enlem ve boylamdan eşsiz bi id elde ettik

train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")

train_set, test_set = train_test_split(housing, test_size=0.2,
                                       random_state=42)  # bunula direk headpladı yukardsakiler gereksiz oldu

housing["income_cat"] = pd.cut(housing["median_income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])
# burda median median_income göre katagöri ettik ve income_cat diye yine artibutes oluşturarak buraya veriyi ekledi.


# 67 den intibaren yapılan işlemlerde cut fonkisyonu ve parametreleri kullanılarak
# böleceği değer aralığını belirledik ve buna label verdik

housing["income_cat"].hist()  # yaptığımız yeni gruplandırmayı göstermek için hist grafiği oluşturduk

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)  # ayarlarını yaptı karıştırma oranı falan
for train_index, test_index in split.split(housing, housing[
    "income_cat"]):  # income_cat'e göre grupladı sonra karıştırılmamış veriyi karıştırdı
    strat_train_set = housing.loc[
        train_index]  # index dediği şey aslında bir id. train_index printleyine bize valuların idsini dönücek
    strat_test_set = housing.loc[test_index]  # yukardakinin aynısı sadece traine göre
# loc komutu ile etiket kullananarak verimize ulaşırken, iloc komutunda satır ve       iloc dizinin varsayılan indexini alır loc ise dizinin içindeki idlere göre alır
# sütün index numarası ile verilerimize ulaşmaktayız, Yani loc komutunu kullanırken
# satır yada kolon ismi belirtirken, iloc komutunda satır yada sütünün index numarasını belirtiyoruz.

print(strat_test_set["income_cat"].value_counts() / len(
    strat_test_set))  # labellarda dağılım ornını verdi labellarda income_cat label ı
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1,
              inplace=True)  # burada drop ile atributes ve ya id lerle satır ve ya kolonları siliyor

housing = strat_train_set.copy()
housing.plot(kind="scatter", x="longitude", y="latitude")
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"] / 100, label="population", figsize=(10, 7),
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
             )
plt.legend()  # grafiğin üstündeki tanıtan kutucuk

corr_matrix = housing.corr()  # corolasyon matrix

corr_matrix["median_house_value"].sort_values(
    ascending=False)  # corolesyon matrixden median ı seçti sonra o değerleri ters sıraladı

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]

scatter_matrix(housing[attributes], figsize=(
    12, 8))  # artibutlar arasındaki korelasyomlar check etmek için scatter_matrix kullanarak grafik oluşturduk

housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)

housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]

corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

#..................................................week_3_başlangıç.................................................................

housing = strat_train_set.drop("median_house_value", axis=1)  # yukarıda tanımladığımız train set kolonu silmek için
housing_labels = strat_train_set["median_house_value"].copy()  # kopyaladı

housing.dropna(subset=["total_bedrooms"])  # option 1
housing.drop("total_bedrooms", axis=1)  # option 2
median = housing["total_bedrooms"].median()  # option 3
# bunların hepsi aynı şeyi  yani siler
housing["total_bedrooms"].fillna(median, inplace=True)  # na = not available bu da siler

imputer = SimpleImputer(
    strategy="median")  # datasetteki boş verileri ortalaması nı alıp bu ortalamaları boş değerlere atar

housing_num = housing.drop("ocean_proximity", axis=1)

imputer.fit(housing_num)  # bu bir class boş değerleri doldurmaya yarar

print(imputer.statistics_)  # medianları yazdırdı alttakide aynı işi yaptı

print(housing_num.median().values)  # median ı hesaplayıp değerleri yazdırdık

X = imputer.transform(housing_num)  # housing_num içindeki tüm boş değerleri empoze eder

housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)  # pandas dataframe i oluşturuduk

housing_cat = housing[["ocean_proximity"]]
print(housing_cat.head)

ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(
    housing_cat)  # modelin verimliliğini artırmak için fit ve transformun birleştirilmiş halidir
print(housing_cat_encoded[:10])

print(ordinal_encoder.categories_)

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
print(housing_cat_1hot)
housing_cat_1hot.toarray()  # array e çevirir
print(cat_encoder.categories_)

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):  # no args or *kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


# model tanımladık baseestimeter ve transformermixin kullanarrak iki tane yeni kolon üretttik


attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)  # yukardaki classı kullandık
housing_extra_attribs = attr_adder.transform(housing.values)
#
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])  # bu tarz transformationların sıralı yapılmasında yardımcı olur
housing_num_tr = num_pipeline.fit_transform(housing_num)

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])  #Bu tahmin edici, girişin farklı sütunlarının veya sütun alt kümelerinin ayrı ayrı dönüştürülmesine ve her transformatör tarafından oluşturulan özelliklerin
#tek bir özellik alanı oluşturmak için birleştirilecektir.

housing_prepared = full_pipeline.fit_transform(housing)
