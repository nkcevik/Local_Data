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
path = path + "/data/cikti.xlsx"
df = pd.read_excel(path)
df['CNTRL'] = pd.Series(np.zeros(len(df)), index=df.index)

count = 0
print(df["GRUP"].describe())
for index1 in range(2, len(df)):
        for index2 in range(3, len(df)):
                if index1 == index2:
                        break

                if df["GRUP"].iloc[index1] == df["GRUP"].iloc[index2]:
                        df["CNTRL"].iloc[index2] = 1
                        df["CNTRL"].iloc[index1] = 1
 


list_of_deleted_index = df[df["CNTRL"] != 1].index
print(len(list_of_deleted_index))

for index in list_of_deleted_index:
        df = df.drop(index)

df.sort_values("GRUP", axis = 0, ascending = True, 
                 inplace = True, na_position ='last') 


path = os.path.dirname(__file__)
path , file = os.path.split(path)
path = path + "/data/cikti_grup.xlsx"
print("xx")
df.to_excel(path)

