import numpy as np
import pandas as pd

# 5x2 NumPy array oluşturduk
my_data = np.array([[0, 3], [10, 7], [20, 9], [30, 14], [40, 15]])

# İki sütunun adlarını tutan bir Python listesi oluşturuk.
my_column_names = ['temperature', 'activity']

# DataFrame
my_dataframe = pd.DataFrame(data=my_data, columns=my_column_names)


print(my_dataframe)

# activity kolonuna 2 ekleyerek yeni bir adjusted kolonu oluşturduk
my_dataframe["adjusted"] = my_dataframe["activity"] + 2

print(my_dataframe)

print("Rows #0, #1, and #2:")
print(my_dataframe.head(3), '\n')

print("Row #2:")
print(my_dataframe.iloc[[2]], '\n')

print("Rows #1, #2, and #3:")
print(my_dataframe[1:4], '\n')

print("Column 'temperature':")
print(my_dataframe['temperature'])




"""Task 1: Create a DataFrame
Do the following:

Create an 3x4 (3 rows x 4 columns) pandas DataFrame in which the columns are named Eleanor, Chidi, Tahani, and Jason. Populate each of the 12 cells in the DataFrame with a random integer between 0 and 100, inclusive.

Output the following:

the entire DataFrame
the value in the cell of row #1 of the Eleanor column
Create a fifth column named Janet, which is populated with the row-by-row sums of Tahani and Jason.

To complete this task, it helps to know the NumPy basics covered in the NumPy UltraQuick Tutorial."""


my_matrix = np.random.randint(low=1, high=101, size=(3, 4))
my_column_names2 = ['Eleanor', 'Chidi', 'Tahani', 'Jason']
my_dataframe2 = pd.DataFrame(data=my_matrix, columns=my_column_names2)
print(my_dataframe2, '\n')
print(my_dataframe2.Eleanor[1])
my_dataframe2["Janet"] = my_dataframe2["Tahani"] + my_dataframe2["Jason"]
print(my_dataframe2)



