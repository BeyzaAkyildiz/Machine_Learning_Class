import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from zlib import crc32
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix



def load_housing_data():
    csv_path = "../input/housing.csv"
    return pd.read_csv(csv_path)


housing = load_housing_data()  # housing e dataseti attı
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

