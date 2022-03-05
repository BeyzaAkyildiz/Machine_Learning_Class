from sklearn import datasets

import matplotlib.pyplot as plt

# Load the digits dataset
digits = datasets.load_digits()

# Display the last digit
plt.figure(1, figsize=(3, 3))
plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation="nearest") # datasetdeki en sonuncu elemanı
# getirdi yani 8 sayısının digitlerini getrdi
plt.imshow(digits.images[2], cmap=plt.cm.gray_r, interpolation="nearest")
plt.imshow(digits.images[3], cmap=plt.cm.gray_r, interpolation="nearest")
plt.imshow(digits.images[4], cmap=plt.cm.gray_r, interpolation="nearest")
plt.imshow(digits.images[11], cmap=plt.cm.gray_r, interpolation="nearest")
print(len(digits.images)) # 1797  tane resim var
plt.show()
# bu dataset pixel tutar