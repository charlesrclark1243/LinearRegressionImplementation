from LinReg import LinReg
from sklearn import datasets, metrics
import numpy as np

X, y, coeffs = datasets.make_regression(n_samples=10000, n_features=1, n_informative=1, bias=0.0, noise=0.0, coef=True, random_state=42)

X_train, y_train = X[:8500], y[:8500]
X_test, y_test = X[8500:], y[8500:]

model = LinReg()
model.fit(X_train, y_train, bias=False)

print(f'Model Estimated Coefficients: {model.coefficients}')
print(f'Actual Coefficients: {coeffs}')

y_pred = model.predict(X_test)
print(f'Model R2 Score: {metrics.r2_score(y_test, y_pred)}\n')

X, y, coeffs = datasets.make_regression(n_samples=10000, n_features=1, n_informative=1, bias=2.35, noise=0.0, coef=True, random_state=84)

X_train, y_train = X[:8500], y[:8500]
X_test, y_test = X[8500:], y[8500:]

model = LinReg()
model.fit(X_train, y_train, bias=True)

print(f'Model Estimated Bias: {model.bias}')
print(f'Model Estimated Coffiecients: {model.coefficients}')
print(f'Actual Coefficients: {coeffs}')

y_pred = model.predict(X_test)
print(f'Model R2 Score: {metrics.r2_score(y_test, y_pred)}\n')

X, y, coeffs = datasets.make_regression(n_samples=10000, n_features=3, n_informative=3, bias=10, noise=0.0, coef=True, random_state=84)

X_train, y_train = X[:8500], y[:8500]
X_test, y_test = X[8500:], y[8500:]

model = LinReg()
model.fit(X_train, y_train, bias=True)

print(f'Model Estimated Bias: {model.bias}')
print(f'Model Estimated Coffiecients: {np.reshape(model.coefficients, (1, len(model.coefficients)))}')
print(f'Actual Coefficients: {coeffs}')

y_pred = model.predict(X_test)
print(f'Model R2 Score: {metrics.r2_score(y_test, y_pred)}\n')

X, y, coeffs = datasets.make_regression(n_samples=10000, n_features=3, n_informative=3, bias=10, noise=10.0, coef=True, random_state=126)

X_train, y_train = X[:8500], y[:8500]
X_test, y_test = X[8500:], y[8500:]

model = LinReg()
model.fit(X_train, y_train, bias=True)

print(f'Model Estimated Bias: {model.bias}')
print(f'Model Estimated Coffiecients: {np.reshape(model.coefficients, (1, len(model.coefficients)))}')
print(f'Actual Coefficients: {coeffs}')

y_pred = model.predict(X_test)
print(f'Model R2 Score: {metrics.r2_score(y_test, y_pred)}\n')

X, y, coeffs = datasets.make_regression(n_samples=10000, n_features=3, n_informative=3, bias=10, noise=100.0, coef=True, random_state=84)

X_train, y_train = X[:8500], y[:8500]
X_test, y_test = X[8500:], y[8500:]

model = LinReg()
model.fit(X_train, y_train, bias=True)

print(f'Model Estimated Bias: {model.bias}')
print(f'Model Estimated Coffiecients: {np.reshape(model.coefficients, (1, len(model.coefficients)))}')
print(f'Actual Coefficients: {coeffs}')

y_pred = model.predict(X_test)
print(f'Model R2 Score: {metrics.r2_score(y_test, y_pred)}\n')

X, y, coeffs = datasets.make_regression(n_samples=10000, n_features=3, n_informative=3, bias=10, noise=100.0, coef=True, random_state=84)

X_train, y_train = X[:8500], y[:8500]
X_test, y_test = X[8500:], y[8500:]

model = LinReg()
model.fit(X_train, y_train, bias=True)

print(f'Model Estimated Bias: {model.bias}')
print(f'Model Estimated Coffiecients: {np.reshape(model.coefficients, (1, len(model.coefficients)))}')
print(f'Actual Coefficients: {coeffs}')

y_pred = model.predict(X_test)
print(f'Model R2 Score: {metrics.r2_score(y_test, y_pred)}\n')