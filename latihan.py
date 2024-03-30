from sklearn.metrics import mean_squared_error, r2_score
from plot import Plot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

class Latihan():
    def __init__(self, correlation, df):
        print(f'corr {correlation}')
        self.correlation = correlation
        self.df = df
        self.high_corr_features = correlation.nlargest(n=3, columns='households').index[1:].tolist()
        
    def evaluate_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print("Mean Squared Error:", mse)
        print("R-squared:", r2)
        return y_pred
    
    def exercise_two(self):
        print('latihan 2 \n')
        print(self.high_corr_features)
        X = self.df[self.high_corr_features]
        # Ekstrak variabel dependen
        y = self.df['households']
        # Pembagian data
        # Bagi dataset menjadi training dan test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        
        # Model Regresi Linear Multiple
        print('\nMultiple Linear Regression untuk memprediksi jumlah rumah tangga:')
        # Buat model
        multiple_lr = LinearRegression()

        # Latih model
        multiple_lr.fit(X_train, y_train)
        plot = Plot()
        # Evaluasi model
        y_pred = self.evaluate_model(multiple_lr, X_test, y_test)
        plot.plotMultiple(y_test, y_pred)
    
    def exercise_three(self):
        print('latihan 3 \n')
        
        # Split data
        X = self.df.drop('median_house_value', axis=1)  # Independent variables
        y = self.df['median_house_value']  # Dependent variable
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Evaluate performance for different kernels
        kernels = ['linear', 'poly', 'rbf']
        for kernel in kernels:
            print(f'\nKernel: {kernel}')
            svr_model = SVR(gamma='scale', kernel=kernel)
            svr_model.fit(X_train, y_train)
            y_pred = self.evaluate_model(svr_model, X_test, y_test)