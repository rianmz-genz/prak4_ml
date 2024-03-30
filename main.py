import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from dataset import Dataset
from latihan import Latihan
# Impor dataset
newDataset = Dataset('src/housing.csv')
df = newDataset.df
dataset = newDataset.dataset

# Cek informasi dataset
print(df.info())

# Hitung jumlah missing value per kolom
mv = df.isna().sum()
print('\nJumlah missing value tiap kolom:\n', mv)

# Mengatasi data kategorikal:
# Ubah kolom 'ocean_proximity' dengan LabelEncoder
label_encoder_x = LabelEncoder()
df['ocean_proximity'] = label_encoder_x.fit_transform(df['ocean_proximity'])

# Korelasi antar variabel
correlation = df.corr(method='pearson')
# Mengatasi missing value:
# Isi missing value dengan median

# Bagi dataframe menjadi 20 blok berdasarkan 'households'
df['block'] = pd.cut(df['households'], bins=20, labels=False)

# Fungsi untuk mengisi NaN dengan median dari setiap blok
def fill_nan_with_block_median(group):
    return group.fillna(group.median())

# Isi NaN di 'total_bedrooms' dengan median dari bloknya
df['total_bedrooms'] = df.groupby('block')['total_bedrooms'].transform(fill_nan_with_block_median)

# Hapus kolom 'block' setelah mengisi NaN
df.drop(columns=['block'], inplace=True)

# Scatter plot: 'median_income' vs 'median_house_value'
sns.scatterplot(x='median_income', y='median_house_value', data=df)
plt.xlabel('Median Income')
plt.ylabel('Median House Value')
plt.title('Income vs House Value')
plt.show()

# Scaling fitur dengan MinMaxScaler
scaler = MinMaxScaler()
df[['latitude', 'longitude']] = scaler.fit_transform(df[['latitude', 'longitude']])

# Ekstrak variabel independen
X = df.drop(columns=['median_house_value'])

# Ekstrak variabel dependen
y = df['median_house_value']

# Seleksi fitur menggunakan chi-square
# Pilih 7 fitur terbaik
selector = SelectKBest(score_func=chi2, k=7)
X_selected = selector.fit_transform(X, y)

# Pembagian data
# Bagi dataset menjadi training dan test set
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=0)

# Fungsi Evaluasi Model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Mean Squared Error:", mse)
    print("R-squared:", r2)
    return y_pred

# Model Regresi Linear Sederhana
# Buat model
simple_lr = LinearRegression()

# Latih model dengan kolom 'median_income' (X_train)
simple_lr.fit(X_train[:, 5].reshape(-1, 1), y_train)

print('\n')
# Evaluasi model dengan kolom 'median_income' (X_test)
print("Simple Linear Regression:")
y_pred = evaluate_model(simple_lr, X_test[:, 5].reshape(-1, 1), y_test)

# Plot hasil
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Simple Linear Regression: ActualPredicted Values for Median House Value')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.show()

# Regresi Linear

'''
Multiple Linear Regression
'''
print('\n')
#membuat model
multiple_lr = LinearRegression()
#melatih model
multiple_lr.fit(X_train, y_train)
#evaluasi model
print("Multiple Linear Regression:")
y_pred = evaluate_model(multiple_lr, X_test, y_test)
#plot hasil



latihan = Latihan(correlation, df)

latihan.exercise_two()
latihan.exercise_three()
