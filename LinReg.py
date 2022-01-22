import numpy as np

class LinReg:
    def __init__(self):
        self.bias = 0.0
        self.coefficients = []

    def fit(self, X_train, y_train, bias=False):
        if isinstance(X_train, np.ndarray) and isinstance(y_train, np.ndarray) and len(y_train) == len(X_train):
            if y_train.ndim == 1:
                y_train = np.reshape(y_train, (len(y_train), 1))
            
            if X_train.ndim == 1 or X_train.shape[1] == 1:
                X_train = np.reshape(X_train, (len(X_train), 1))

                Xy_train = X_train * y_train
                Xy_train_avg = np.average(Xy_train)

                X_train_avg = np.average(X_train)
                y_train_avg = np.average(y_train)

                slope_numerator = Xy_train_avg - (X_train_avg * y_train_avg)

                X_train_sqr = np.square(X_train)
                X_train_sqr_avg = np.average(X_train_sqr)

                X_train_avg_sqr = np.square(X_train_avg)

                slope_denominator = X_train_sqr_avg - X_train_avg_sqr

                slope = slope_numerator / slope_denominator
                self.coefficients = np.array([slope], ndmin=2)

                if (bias):
                    self.bias = y_train_avg - (slope * X_train_avg)
            else:
                if (bias):
                    ones = np.ones((len(X_train), 1))
                    X_train = np.concatenate((ones, X_train), axis=1)

                component_one = np.matmul(X_train.T, X_train)
                component_one_inv = np.linalg.inv(component_one)

                component_two = np.matmul(X_train.T, y_train)

                all_coeffs = np.matmul(component_one_inv, component_two)
                
                if (bias):
                    self.bias = all_coeffs[0, 0]
                    self.coefficients = all_coeffs[1:, 0]
                else:
                    self.coefficients = all_coeffs

                self.coefficients = np.reshape(self.coefficients, (len(self.coefficients), 1))
        else:
            raise ValueError(f'Fitting Error')
    
    def predict(self, X_test):
        if isinstance(X_test, np.ndarray):
            
            y_pred = np.empty((len(X_test), 1))
            y_pred.fill(self.bias)

            y_pred = y_pred + np.matmul(X_test, self.coefficients)
            return np.reshape(y_pred, (len(X_test), 1))
        else:
            raise ValueError(f'Predicting Error')
