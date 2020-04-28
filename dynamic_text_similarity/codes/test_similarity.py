import pandas as pd 
import numpy as np
import Levenshtein as lv
import re, math
from collections import Counter
import os
import sys
from text_similarity import Similarity as sm
import time

class TestSimilarity:
    # Initializer / Instance Attributes
    def __init__(self):
        self.accuracy_score = 90
        self.similarity_algorithm = "all"
        print("come")

    def test_eti(self):
        path = os.path.dirname(__file__)
        path , file = os.path.split(path)
        file = "eti_data"
        path = path + "/data/eti_data/Muhtp_Spart.xlsx"
        dataset = pd.read_excel(path) 

        # indexj_list = ["KAYIT SIRASI"]
        dataset["INDEX_NO"] = dataset["KAYIT SIRASI"]

        # column_names = ["MANDT","NAME1"]
        dataset["DATA"] = dataset["NAME1"].map(str)  + "_" +  dataset["MRTNM"].map(str)  + "_" +  dataset["BLTNM"].map(str) 

        # DATAFRAME
        dataset = dataset[["DATA", "INDEX_NO"]]
        # dataset = dataset.head(100)

        # call and fit Similarity class
        ts = sm(self.similarity_algorithm, self.accuracy_score, "DATA", "INDEX_NO", dataset, file)
        ts.fit()

    def test_mlz_active(self):
        path = os.path.dirname(__file__)
        path , file = os.path.split(path)
        file = "mlz_active"
        path = path + "/data/mlz_active/örnek_malzeme_verisi_active.xlsx"
        dataset = pd.read_excel(path) 

        # index_join_list = ['SYSTEM','MLZNO']
        dataset["INDEX_NO"] = dataset["SYSTEM"].map(str) + "_" + dataset["MLZNO"].map(str) 
        
        # column_names = ["MLZ_KISA_TANIM_TR"]
        # dataset.rename(columns={'MLZ_KISA_TANIM_TR': 'DATA'}, inplace=True)
        dataset["DATA"] = dataset["MLZ_KISA_TANIM_TR"].map(str)  


        # DATAFRAME
        dataset = dataset[["DATA", "INDEX_NO"]]
        # dataset = dataset.head(100)

        # call and fit Similarity class
        ts = sm(self.similarity_algorithm, self.accuracy_score, "DATA", "INDEX_NO", dataset, file)
        ts.fit()

    def test_musteri_mali_alan(self):
        path = os.path.dirname(__file__)
        path , file = os.path.split(path)
        file = "musteri_mali_alan"
        path = path + "/data/musteri_mali_alan/MUSTERI_MALI_ALAN_KONSOLIDE_BENZERLIK.xlsx"
        dataset = pd.read_excel(path)

        # index_join_list = ['SYSTEM','MUSTERI_NO']
        dataset["INDEX_NO"] = dataset["SYSTEM"].map(str)  + "_" +  dataset["MUSTERI_NO"].map(str) 

        # column_names = ["MUSTERI_AD","VERGI_NO"]
        dataset["DATA"] = dataset["MUSTERI_AD"].map(str)  + "_" +  dataset["VERGI_NO"].map(str) 

        # DATAFRAME
        dataset = dataset[["DATA", "INDEX_NO"]]
        # dataset = dataset.head(100)

        # call and fit Similarity class
        ts = sm(self.similarity_algorithm, self.accuracy_score, "DATA", "INDEX_NO", dataset, file)
        ts.fit()

    def test_musteri_tedarikci(self):
        path = os.path.dirname(__file__)
        path , file = os.path.split(path)
        file = "musteri_tedarikci"
        path = path + "/data/musteri_tedarikci/Musteri_Tedarikci_Testdata.xlsx"
        dataset = pd.read_excel(path) 


        # index_join_list = ['SYSTEM','MUSTERI_NO']
        dataset["INDEX_NO"] = dataset["Sistem"].map(str)  + "_" +  dataset["Sistem Kodu"].map(str) 

        # column_names = ["MUSTERI_AD","VERGI_NO"]
        dataset["DATA"] = dataset["Müşteri ADI"].map(str) + "_" + dataset["Vergi No"].map(str) 

        # DATAFRAME
        dataset = dataset[["DATA", "INDEX_NO"]]
        # dataset = dataset.head(100)

        # call and fit Similarity class
        ts = sm(self.similarity_algorithm, self.accuracy_score, "DATA", "INDEX_NO", dataset, file)
        ts.fit()


baslangıc_zaman = time.time()

# ts = TestSimilarity()
# ts.test_eti()

# ts = TestSimilarity()
# ts.test_mlz_active()


# ts = TestSimilarity()
# ts.test_musteri_mali_alan()

ts = TestSimilarity()
ts.test_musteri_tedarikci()

bitis_zaman = time.time()
print(bitis_zaman - baslangıc_zaman)