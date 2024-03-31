from sklearn.metrics import mean_squared_error, r2_score
from plot import Plot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
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
        print('\n')
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
        print('\n')
        print('latihan 3 \n')
        
        # Split data
        X = self.df.drop('median_house_value', axis=1)  # Independent variables
        y = self.df['median_house_value']  # Dependent variable
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Evaluate performance for different kernels
        kernels = ['linear', 'poly', 'rbf']
        for kernel in kernels:
            print(f'\nKernel: {kernel}')
            svr_model = SVR(gamma='auto', kernel=kernel)
            svr_model.fit(X_train, y_train)
            y_pred = self.evaluate_model(svr_model, X_test, y_test)
    
    def exercise_four(self, X_train, y_train, X_test, y_test):
        print('\n')
        print('latihan 4 \n')
        
        # 4. Latih model regresi
        # 4.1 Polynomial Regression
        poly_reg = PolynomialFeatures(degree=2)
        X_train_poly = poly_reg.fit_transform(X_train)
        X_test_poly = poly_reg.transform(X_test)

        # 4.2 SVR
        svr_reg = SVR(kernel='rbf', gamma='auto')
        svr_reg.fit(X_train, y_train)

        # 4.3 Multiple Linear Regression
        mlr_reg = LinearRegression()
        mlr_reg.fit(X_train, y_train)
        
        # 5. Evaluasi model
        # 5.1 Polynomial Regression
        # Prediction
        lin_reg = LinearRegression()
        lin_reg.fit(X_train_poly, y_train)
        poly_pred = lin_reg.predict(X_test_poly)

        # Evaluation (similar to your original code)
        poly_mse = mean_squared_error(y_test, poly_pred)
        poly_r2 = r2_score(y_test, poly_pred)

        # 5.2 SVR
        svr_pred = svr_reg.predict(X_test)
        svr_mse = mean_squared_error(y_test, svr_pred)
        svr_r2 = r2_score(y_test, svr_pred)

        # 5.3 Multiple Linear Regression
        mlr_pred = mlr_reg.predict(X_test)
        mlr_mse = mean_squared_error(y_test, mlr_pred)
        mlr_r2 = r2_score(y_test, mlr_pred)
        
        # 6. Bandingkan hasil
        print('--- Hasil Evaluasi ---')
        print('Model', 'MSE', 'R-squared')
        print('Polynomial Regression', poly_mse, poly_r2)
        print('SVR', svr_mse, svr_r2)
        print('Multiple Linear Regression', mlr_mse, mlr_r2)

        # 7. Kesimpulan
        best_model = 'Multiple Linear Regression'
        if poly_mse < mlr_mse and poly_r2 > mlr_r2:
            best_model = 'Polynomial Regression'
        elif svr_mse < mlr_mse and svr_r2 > mlr_r2:
            best_model = 'SVR'
            
        print('Model terbaik:', best_model)