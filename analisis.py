import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib

# Membuat data
data = pd.DataFrame({
    "Date": ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04", "2022-01-05",
            "2022-01-06", "2022-01-07", "2022-01-08", "2022-01-09", "2022-01-10",
            "2022-01-11", "2022-01-12", "2022-01-13", "2022-01-14", "2022-01-15",
            "2022-01-16", "2022-01-17", "2022-01-18", "2022-01-19", "2022-01-20"],
    "Product": ["Product A", "Product B", "Product C", "Product D", "Product E",
               "Product F", "Product G", "Product H", "Product I", "Product J",
               "Product A", "Product B", "Product C", "Product D", "Product E",
               "Product F", "Product G", "Product H", "Product I", "Product J"],
    "Category": ["Electronics", "Furniture", "Electronics", "Apparel", "Electronics",
                "Furniture", "Apparel", "Electronics", "Furniture", "Apparel",
                "Electronics", "Furniture", "Electronics", "Apparel", "Electronics",
                "Furniture", "Apparel", "Electronics", "Furniture", "Apparel"],
    "Sales": [2500, 1800, 3200, 1400, 2800, 2100, 1600, 2900, 1700, 1900,
             2600, 1900, 3100, 1500, 2700, 2200, 1700, 3000, 1800, 2000]
})

# Mencetak data
print("Data:")
print(data.head())

# Konversi kolom 'Date' menjadi format datetime
data['Date'] = pd.to_datetime(data['Date'])

# Konversi kolom 'Category' menjadi tipe data kategorikal
data['Category'] = data['Category'].astype('category')

# Membuat kolom baru 'Sales_per_Quantity' sebagai hasil pembagian antara 'Sales' dan 'Quantity'
data['Sales_per_Quantity'] = data['Sales'] / 1  # Asumsi Quantity = 1

# Normalisasi kolom 'Sales'
scaler = StandardScaler()
data['Sales_Normalized'] = scaler.fit_transform(data[['Sales']])

# Mencetak hasil setelah transformasi data
print("Data types after transformations:")
print(data.dtypes)
print("Data after adding 'Sales_per_Quantity' and normalizing 'Sales':")
print(data.head())

# Lakukan analisis dan visualisasi data sesuai kebutuhan
print("Descriptive statistics:")
print(data.describe())

sns.histplot(data=data, x="Sales", bins=20)
plt.title("Histogram Penjualan")
plt.xlabel("Penjualan")
plt.ylabel("Frekuensi")
plt.show()

sns.scatterplot(data=data, x="Date", y="Sales")
plt.title("Scatter Plot Penjualan Berdasarkan Tanggal")
plt.xlabel("Tanggal")
plt.ylabel("Penjualan")
plt.xticks(rotation=90)
plt.show()

