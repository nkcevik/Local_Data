import pandas as pd
import os
import sys
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB 
from sklearn.metrics import confusion_matrix

# csv dosyamızı okuduk.
path = os.path.dirname(__file__)
path , file = os.path.split(path)
path = path + "/data/Iris.csv"
data = pd.read_csv(path)

# Bağımlı Değişkeni ( species) bir değişkene atadık
species = data.iloc[:,-1:].values

# Veri kümemizi test ve train şekinde bölüyoruz
x_train, x_test, y_train, y_test = train_test_split(data.iloc[:,1:-1],species,test_size=0.33,random_state=0)

# MultinomialNB sınıfından bir nesne ürettik
mnb = MultinomialNB()

# Makineyi eğitiyoruz
mnb.fit(x_train, y_train.ravel())

# Test veri kümemizi verdik ve iris türü tahmin etmesini sağladık
result = mnb.predict(x_test)

# Karmaşıklık matrisi
cm = confusion_matrix(y_test,result)
print(cm)

# Başarı Oranı
accuracy = accuracy_score(y_test, result)

# Sonuç : 0.96
# 16 + 19 + 16 + 2= 50 tane veri içinden 48 tanesini doğru tahmin edilirken 2 tanesi yanlış tahmin edilmiştir.
# Başarı oranı 48 / 50= 0,96 ‘ dır.
print(accuracy)