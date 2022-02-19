"""DICTIONARY
Why do we need dictionary?

It has 'key' and 'value'
Faster than lists
What is key and value. Example:
dictionary = {'spain' : 'madrid'}
Key is spain.
Values is madrid.

It's that easy.
Lets practice some other properties like keys(), values(), update, add, check, remove key, remove all entries and remove dicrionary."""

# create dictionary and look its keys and values
import numpy as np
import pandas as pd

dictionary = {'spain': 'madrid', 'usa': 'vegas'}
print(dictionary.keys())
print(dictionary.values())

# Keys have to be immutable objects like string, boolean, float, integer or tubles
# List is not immutable
# Keys are unique
dictionary['spain'] = "barcelona"  # update existing entry
print(dictionary)
dictionary['france'] = "paris"  # Add new entry
print(dictionary)
del dictionary['spain']  # remove entry with key 'spain'
print(dictionary)
print('france' in dictionary)  # check include or not
dictionary.clear()  # remove all entries in dict
print(dictionary)

dictionary['spain'] = "hee"
print(dictionary)
dictionary['spain'] = "heett"
print(dictionary)
print('spainn' in dictionary)
print(dictionary)
dictionary['luluu'] = "wiwi"
del dictionary['spain']
print(dictionary)

dictionary.clear()
print(dictionary)

"""PANDAS
What do we need to know about pandas?

CSV: comma - separated values değerler tablodaki virgülle ayrılmıştır bir
dosya formatıdır"""

data = pd.read_csv('inputs/pokemon.csv')  # csv dosyasını okuyup içeriği dataya aktardık

series = data['Defense']
# data['Defense'] = series defense future(kolonu) ın içindeki verileri serise attı dizi oldu
print(type(series))
# <class 'pandas.core.series.Series'> komple series dizisinin türünü yazdırıyor içindekilerin tek tek yazdırmaz

data_frame = data[['Defense']]  # data[['Defense']] = data frame
print(type(data_frame))
# <class 'pandas.core.frame.DataFrame'>
print(series)  # series defense in içindeki veri dizisi
print(data_frame)  # frame series i kapsar
# data[defense :[ aa,bb], attack: [ccc.]]
# [aa,bb] -----> series
# [defence : [aa,bb]] ---->frame
print(type(data_frame.Defense))  # bunu demekle series demek aynı şey series e eriştik

# Comparison operator
print(3 > 2)
print(3 != 2)
# Boolean operators
print(True and False)
print(True or False)

# 1 - Filtering Pandas data frame
x = data['Defense'] > 200  # There are only 3 pokemons who have higher defense value than 200
print(x)  # bunu yazdığımız zaman yukardaki koşulu true ve false olarak dizi şeklinde yazdırdı
print(data[x])  # yukardaki koşulu sağlayanları artibute ve indeksleriyle yazdırdı komple satırıyla yazdırdı yani

# 2 - Filtering pandas with logical_and
# There are only 2 pokemons who have higher defence value than 2oo and higher attack value than 100
print(data[np.logical_and(data['Defense']>200, data['Attack']>100 )])
#print(data[(data['Defense'] > 200) and (data['Attack'] > 100)])  bu kullanım dizi olduğu için and i çalıştıramadı hatalı"""
# bu and bool olduğu için sadece 0 ve 1 olarak tuttuu için yanlış oldu

# This is also same with previous code line. Therefore we can also use '&' for filtering.
print(data[(data['Defense']>200) & (data['Attack']>100)]) # and normalde iki tane işaret yapınca oluyordu burda tek yaptığı için çalışmış
# bu işaret bitwise olduğu için doğru çalıştı

print(0b1010 & 0b0011) # dönen ceap 0010 yani(2) bu bitwise tek tek bit olark hesapladı 1 1 olursa 1 oluyor ama and kullansaydık direk 0 dönicekti

# Stay in loop if condition( i is not equal 5) is true
i = 0
while i != 5 :
    print('i is: ',i)
    i +=1
print(i,' is equal to 5') #0 dan 4 e kadar çalıştı



# Stay in loop if condition( i is not equal 5) is true
lis = ['a','b',3,4,5]
list2 = {'a':5,'b':7,'c':17} # bu dictionary normalde dict de index a,b,c
for i in lis:
    print('i is: ',i)    # burda i index değil indexdeki value değeridir
print('')

# Enumerate index and value of list
# index : value = 0:1, 1:2, 2:3, 3:4, 4:5
for index, value in enumerate(list2): # böyle kullanıldığında enumerate ile index
    # değerleri 0 dan başlayarak verildi normalde a,b,c ydi index i de yazdırmak istediğimiz için enumurate kullandık
    #eğer liste olan lis i de yazdırmak isteseydeik ve index ininde yazdırıcaksak enumurate ile kullanmalıyız
    print(index," : ",value)
print('')

for index, value in enumerate(lis):

    print(index," : ",value)
print('')


# For dictionaries
# We can use for loop to achive key and value of dictionary. We learnt key and value at dictionary part.
dictionary = {'spain':'madrid','france':'paris'}
for key,value in dictionary.items(): # .items() ile normal keyleri döndürür kendisi bir değer vermiyor a,b,c yapıyor kendi keylerini alır
    print(key," : ",value)
print('')

# For pandas we can achieve index and value
for index,value in data[['Attack']][0:1].iterrows(): # .iterrows  pandas dataframini for ile yazdırmak için kullanılır  ????
    print(index," : ",value)
print(' ')






