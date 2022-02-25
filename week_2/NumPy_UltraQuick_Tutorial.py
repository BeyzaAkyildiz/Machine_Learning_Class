import numpy as np

one_dimensional_array = np.array([1.2, 2.4, 3.5, 4.7, 6.1, 7.2, 8.3, 9.5])
print(one_dimensional_array)
#tek boyutlu array



two_dimensional_array = np.array([[6, 5], [11, 7], [4, 8]])
print(two_dimensional_array)
#iki boyutlu array


sequence_of_integers = np.arange(5, 12)
print(sequence_of_integers)

#5 den 12 ye kadar olanları yazdırdı

random_integers_between_50_and_100 = np.random.randint(low=50, high=101, size=(6))
print(random_integers_between_50_and_100)

#random olarak 50 den başlayıp 101 e kadar olan sayılardan 6 tane yazdırdı

random_floats_between_0_and_1 = np.random.random([6])
print(random_floats_between_0_and_1)

# bu sefer 0 dan 1 aralığında altı tane yazdırdı ama tam değil


random_floats_between_2_and_3 = random_floats_between_0_and_1 + 2.0
print(random_floats_between_2_and_3)

# 2 den 3 e kadar olan  tam olmayan sayıları öncekinde 2 toplayarak yazdırdı



random_integers_between_150_and_300 = random_integers_between_50_and_100 * 3
print(random_integers_between_150_and_300)


# sayılar 150 ve 300 aralığında tam sayı 50 ile 100 aralığında random sayı üretip bunu 3 le çarparak elde etti

"""Task 1: Create a Linear Dataset
Your goal is to create a simple dataset consisting of a single feature and a label as follows:

Assign a sequence of integers from 6 to 20 (inclusive) to a NumPy array named feature.
Assign 15 values to a NumPy array named label such that:"""
# 6 dan 20 dahil sayıları yazdır
feature = np.arange(6,21)
print(feature)
label = (feature * 3) + 4
print(label)

"""Task 2: Add Some Noise to the Dataset
To make your dataset a little more realistic, insert a little random noise into each element of the label array you already created. To be more precise, modify each value assigned to label by adding a different random floating-point value between -2 and +2.

Don't rely on broadcasting. Instead, create a noise array having the same dimension as label."""
# burda 2 ila -2 arasında sayılar istiyoruz fakat int istemiyoruz
noise = (np.random.random([15]*4)-2)
print(noise)
label = label + noise
print(label)

