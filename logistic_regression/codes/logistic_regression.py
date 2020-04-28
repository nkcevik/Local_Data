# csv dosyalarını okumak için
import pandas as pd
import numpy as np
import sys
import os

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# csv dosyamızı okuduk.
path = os.path.dirname(__file__)
path , file = os.path.split(path)
path = path + "/data/train.csv"

data = pd.read_csv(path)


# Age kolonundaki değerleri bir değişkene atadık
age = data.iloc[:, 5:6].values

# Bağımlı Değişkeni (sex) bir değişkene atadık
sex  = data.iloc[:,4:5].values

# Age kolonundaki eksik değerlerin yerine tüm kolonun ortalaması yazıldı.

imputer = SimpleImputer(missing_values=np.nan ,strategy="mean")
imputer = imputer.fit(age[: , 0:1])
age[: , 0:1] = imputer.transform(age[: , 0:1])

# DataFrame'e dönüştürdük.
dfAge = pd.DataFrame(data = age, index=range(len(age)), columns=['age'])

# concat fonksiyonu ile bağımsız verileri birleştirdik. ( Survived, Pclass, age, SibSp, Parch )
concat = pd.concat([ data.iloc[:,1:3],dfAge,data.iloc[:,6:8]],axis=1)

# Veri kümemizi test ve train şekinde bölüyoruz
x_train, x_test, y_train, y_test = train_test_split(concat,sex,test_size=0.33,random_state=0)

# LogisticRegression sınıfından bir nesne ürettik
lr = LogisticRegression(random_state=0)

# Makinemizi eğittik
lr.fit(x_train,y_train)

# Test veri kümemizi verdik ve cinsiyet tahmin etmesini sağladık
result = lr.predict(x_test)

# Sonuçları ekrana yazdırdık
for i in range(len(result)):
    print("Tahmin : " + result[i] + ", Gerçek Değer : " + y_test[i])


# Başarı Oranı
accuracy = accuracy_score(y_test, result)
print(accuracy)

right = 0
false = 0
for i in range(len(y_test)):
    if y_test[i] == result[i]:
            right = right + 1
    else:
            false = false + 1
            
print("Right: ", right)
print("False: ", false)