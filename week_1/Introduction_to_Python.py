#https://www.kaggle.com/kanncaa1/data-sciencetutorial-for-beginners/

#kütüphanelerin eklenmesi
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

data = pd.read_csv('inputs/pokemon.csv') #(csv) virgüllerle ayrılmı data/değer demek. read_csv fonkisyonu pokemon.cvsyi okur ve
# dataya,ataraslında ismi pandas olan ama pd olarak adlanırılan pd)kütüphanesi içindeki fonksiyona ulaştık

print(data.head())# data verisinin içindeki head ile en baştaki default değer 5 atandığı için ilk 5 değeri yazdırıdk değer verek
# değiştirebilir kaç tanesini göstericeğini bu print yazmadan ekrana yazdırılmadı aşağıdakinide printle yazdırabiliriz ama yazmadan
# da çalışıyor log çıktısı olmadığı için yazmaz
data.info() #az önceki tablonun bilgisini getirdi
print(data.corr())# krolasyon oranını buldu hesap çok önemli değil
f,ax = plt.subplots(figsize=(18, 18))#kutucuğun boyutu 2 tan değer var
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show() #şekli ekrana getirdi
data.head(10)#burda yukardaki gibi ama değer olarak 10 giridiğimiz için 10 satır hetirdi
#print olmadan neden yazmıyor sor???
data.columns #print olunca kolonları yazdırır
# week_2 baslangıç


# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data.Speed.plot(kind = 'line', color = 'g',label = 'Speed',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')#3speed verilerini kullanarak plot fonksiyonunu çalıştırarak tabloyu oluşturdu
data.Defense.plot(color = 'r',label = 'Defense',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')#defence verilerini kullanarak plot fonksiyonunu çalıştırarak tabloyu oluşturdu
plt.legend(loc='upper right')     # legend = puts label into plot  #yukarıdaki speed ve defence labelları sağ köşeye konumlandırdı
plt.xlabel('x axis')   #bu x yazısı            # label = name of label
plt.ylabel('y axis') # bu y yaazısı
plt.title('Line Plot')   #bu diğer yazı          # title = title of plot
plt.show() #hepsini bu yazdıdı ekrana görseli oluşturdu



# Scatter Plot
# x = attack, y = defense
data.plot(kind='scatter', x='Attack', y='Defense',alpha = 0.5,color = 'red')
plt.xlabel('Attack')              # label = name of label
plt.ylabel('Defence')
plt.title('Attack Defense Scatter Plot')
plt.show() #bu yukardaki gibi özellikleri verilen tabloyu ekrana getirdi/gösterdi

# Histogram
# bins = number of bar in figure
data.Speed.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show() #tabloyu ekrana getirdi

# clf() = cleans it up again you can start a fresh


# clf() = cleans it up again you can start a fresh
data.Speed.plot(kind = 'hist',bins = 50) #data içindeki speed sutunu var o sutunu kullanarak o stunun içindekileri plot ileg speed birşey değil tablodaki kolonun adı
#grafie dönüştürdü ama ekrana getirmeden clf sildi
plt.clf()
# We cannot see plot due to clf()




