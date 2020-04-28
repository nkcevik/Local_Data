import pandas as pd 
import numpy as np
import Levenshtein as lv
import re, math
from collections import Counter
import os
import sys
from StyleFrame import StyleFrame, Styler



path = os.path.dirname(__file__)
path , file = os.path.split(path)
file = "musteri_tedarikci"
path = path + "/data/cikti_all_musteri_tedarikci.xlsx"
df = pd.read_excel(path) 
df['CNTRL'] = pd.Series(np.zeros(len(df)), index=df.index)

count = 0
print(df["GRUP_LEV"].describe())
for index1 in range(2, len(df)):
        for index2 in range(3, len(df)):
                if index1 == index2:
                        break

                if df["GRUP_LEV"].iloc[index1] == df["GRUP_LEV"].iloc[index2]:
                        df["CNTRL"].iloc[index2] = 1
                        df["CNTRL"].iloc[index1] = 1
 


list_of_deleted_index = df[df["CNTRL"] != 1].index
print(len(list_of_deleted_index))

for index in list_of_deleted_index:
        df = df.drop(index)

df.sort_values("GRUP_LEV", axis = 0, ascending = True, 
                 inplace = True, na_position ='last') 


path = os.path.dirname(__file__)
path , file = os.path.split(path)
file = "musteri_tedarikci"
path = path + "/data/musteri_tedarikci/cikti_grup_Musteri_Tedarikci.xlsx"

print("xx")
df.to_excel(path)

