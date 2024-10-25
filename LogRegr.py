import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import sklearn
sklearn.set_config(transform_output='pandas')
from sklearn.metrics import r2_score


st.write("""# Исследование логистической регресии""")

file = st.sidebar.file_uploader("Загрузите CSV-файл", type="csv")
if file is not None:
    train = pd.read_csv(file)
    st.write(train.head(5))
else:
    st.stop()

learning_rate = float(st.sidebar.text_input("Введите скорость обучения"))
epochs = int(st.sidebar.text_input("Введите количество эпох обучения"))


X_train, y_train = train.drop('Personal.Loan', axis=1), train['Personal.Loan']

X_train = X_train.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
X_train = X_train.apply(pd.to_numeric, errors='coerce')

norma = StandardScaler()

X_train = norma.fit_transform(X_train)


class LogReg:
    def __init__(self, learning_rate, epochs):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def derivative_w0(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        y_pred = self.predict(X)
        return (y_pred - y)

    def derivative_w1(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        y_pred = self.predict(X)
        return X[:, 0] * (y_pred - y)
        
    def derivative_w2(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        y_pred = self.predict(X)
        return X[:, 1] * (y_pred - y)

    def grad(self, X, y):
        return np.array([self.derivative_w1(X, y), self.derivative_w2(X, y)])

    def fit(self, X, y):
        self.coef_ = np.random.uniform(-1, 1, size=X.shape[1]) 
        self.intercept_ = np.random.uniform(-1, 1)
        #print(f'Стартовые веса {self.coef_}, {self.intercept_}')

        X = np.array(X)
        y = np.array(y)

        for epoch in range(self.epochs): 

            self.coef_ = self.coef_ - self.learning_rate * self.grad(X, y).mean(axis=1)
            self.intercept_ = self.intercept_ - self.learning_rate * self.derivative_w0(X, y).mean()

            #if epoch%10 == 0:
                #print(f'Обновленные веса {self.coef_}, {self.intercept_}')
                #print(f'---' * 20)
    
    def sigmoid(self, z):
        return (1/(1 + np.exp(-z)))

    def predict(self, X):
        y_pred = self.sigmoid(X @ self.coef_ + self.intercept_)
        return y_pred 

    def score(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)
    

my_logreg = LogReg(learning_rate, epochs)

my_logreg.fit(X_train, y_train)


dict_w = {f'Вес для признака {X_train.columns[0]}': round(my_logreg.coef_[0], 5),
        f'Вес для признака {X_train.columns[1]}': round(my_logreg.coef_[1], 5),
        "Свободный член регрессии": round(my_logreg.intercept_, 5)}

st.write("## Результаты с применением логистической регрессии")

train_w = pd.DataFrame(dict_w, index=[1])

st.write(train_w)

st.write("## Сравнение результата с логистической регрессией в sklearn")

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train, y_train)
#lr.coef_, lr.intercept_

dict_w_sk = {f'Вес для признака {X_train.columns[0]}': round(lr.coef_[0][0], 5),
          f'Вес для признака {X_train.columns[1]}': round(lr.coef_[0][1], 5),
          "Свободный член регрессии": round(lr.intercept_[0], 5)}

train_w_sk = pd.DataFrame(dict_w_sk, index=[1])

st.write(train_w_sk)


st.write("## Визуализация распределений двух классов с предсказательной прямой")

button = st.button('Построить scatter plot ')
if button:
    x_min, x_max = - 1.5, 3.5
    y_min, y_max = -2, 2
    xx, yy = np.meshgrid(np.linspace(-1.5, 3.5, 1000), np.linspace(-2, 2, 1000))
    Z = my_logreg.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 6))
    plt.scatter(X_train['CCAvg'], X_train['Income'], c=y_train.map({1: 'red', 0: 'blue'}), label='Данные')

    plt.contour(xx, yy, Z, levels=[0.5], colors='black')
    plt.xlabel( f"{X_train.columns[0]}")
    plt.ylabel( f"{X_train.columns[1]}")
    plt.title("Распределения двух классов с предсказательной прямой")


    st.pyplot(plt)