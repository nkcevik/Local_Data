import pandas as pd 
import numpy as np
import Levenshtein as lv
from Levenshtein import * 
import re, math
from collections import Counter
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from SimilarityAlgorithms import SimilarityAlgorithms
from CleanData import CleanData

WORD = re.compile(r'\w+')

class Similarity:

    # Initializer / Instance Attributes
    @classmethod
    def __init__(self, similarity_algorithm, accuracy_ratio, column, unique_column, dataset):
        self.similarity_algorithm = similarity_algorithm
        self.accuracy_ratio = accuracy_ratio
        self.unique_column = unique_column
        self.column = column
        self.dataset = dataset
        self.grup_column = "GRUP"
        self.score_column= "YUZDE"
        

        self.similarity_names = [ "levenshtein", "cosin", "jaccard", "all"]

    def control_similarity_algorithm(self):    
        for algorithm in self.similarity_names:
            if algorithm == self.similarity_algorithm: 
                print("Algoritma: ", algorithm)
                print("Score: ", self.accuracy_ratio)
                return 0   
        print("Geçersiz bir benzerlik algoritması girdiniz." +
              "\n .")
        return 1
    
    def fit(self):
        # print("I'm also here.")

        # if the name of similarity algorithm is wrong, process is terminated.
        if self.control_similarity_algorithm():
            return 
        # print("I'm here.")
        if self.similarity_algorithm == "all":
            self.all_similarity_algorithm()
        else:
            self.fit_similarity_algorithm()

    def all_similarity_algorithm(self):
        cd = CleanData(self.dataset, self.unique_column, self.column)
        df = cd.all_clean_dataframe()
        path = os.path.dirname(os.path.realpath('__file__')) + "/cikti_all.xlsx"
        algorithm_grup_number = [1, 1, 1]
        algorithm_grup = ["GRUP_LEV", "GRUP_COS", "GRUP_JAC"]
        algorithm_score = ["SCORE_LEV", "SCORE_COS", "SCORE_JAC"]
        for index in range(0, 3):
            df[algorithm_grup[index]].iloc[1] = algorithm_grup_number[index]
            df[algorithm_score[index]].iloc[1] =  "base"

            cntrl = 0
            for index1 in range(2, len(df)):
                algorithm_grup_number[index] += 1
                while df[algorithm_grup[index]].iloc[index1] != 0:
                    if index1 + 1 < len(df):
                        index1 += 1
                    else:
                        cntrl = 1
                        break
                if cntrl == 1:
                    break

                df[algorithm_grup[index]].iloc[index1] = algorithm_grup_number[index]
                df[algorithm_score[index]].iloc[index1] = "base"

                for index2 in range(len(df)):
                    if index1 != index2:
                        if algorithm_grup[index] == "GRUP_LEV":
                            sc = lv.ratio(df[self.column].iloc[index1], df[self.column].iloc[index2]) * 100
                        elif algorithm_grup[index] == "GRUP_COS":
                            sc = self.get_cosine(df[self.column].iloc[index1], df[self.column].iloc[index2]) 
                        elif algorithm_grup[index] == "GRUP_JAC":
                            sc = self.jaccard_similarity(df[self.column].iloc[index1], df[self.column].iloc[index2])

                        if sc > self.accuracy_ratio:
                            if df[algorithm_grup[index]].iloc[index2] == 0:
                                df[algorithm_grup[index]].iloc[index2] = algorithm_grup_number[index]
                                df[algorithm_score[index]].iloc[index2] = round(sc, 2)       


        
        
        print("xx")
        df.to_excel(path)

    def fit_similarity_algorithm(self):
        cd = CleanData(self.dataset, self.unique_column, self.column)
        df = cd.all_clean_dataframe()
        path = os.path.dirname(os.path.realpath('__file__')) + "/cikti.xlsx"
        grup = 1

        df[self.grup_column].iloc[1] = grup
        df[self.score_column].iloc[1] =  "base"
        # print(self.grup_column, " ", 0, "", grup)
        cntrl = 0
        for index1 in range(2, len(df)):
            grup += 1
            while df[self.grup_column].iloc[index1] != 0:
                if index1 + 1 < len(df):
                    # print("hello")
                    index1 += 1
                else:
                    cntrl = 1
                    break
                # print(index1)
            if cntrl == 1:
                break

            df[self.grup_column].iloc[index1] = grup
            df[self.score_column].iloc[index1] = "base"

            for index2 in range(len(df)):
                if index1 != index2:
                    score = self.scores(df[self.column].iloc[index1], df[self.column].iloc[index2])
                    if score > self.accuracy_ratio:
                        if df[self.grup_column].iloc[index2] == 0:
                            df[self.grup_column].iloc[index2] = grup
                            df[self.score_column].iloc[index2] = round(score, 2) 
        print("xx")
        df.to_excel(path)
    
    def jaccard_similarity(self, x, y):
        vec1 = self.text_to_vector(x)
        vec2 = self.text_to_vector(y)

        intersection_cardinality = len(set(vec1).intersection(set(vec2)))
        union_cardinality = len(set(vec1).union(set(vec2)))
        jaccard = intersection_cardinality / float(union_cardinality)

        return jaccard * 100

    def get_cosine(self, x, y):
        vec1 = self.text_to_vector(x)
        vec2 = self.text_to_vector(y)

        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])

        sum1 = sum([vec1[x]**2 for x in vec1.keys()])
        sum2 = sum([vec2[x]**2 for x in vec2.keys()])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)

        if not denominator:
            return 0.0 
        else:
            return (float(numerator) / denominator) * 100
            
    def scores(self, x, y):
        
        # levenshtein    
        if self.similarity_algorithm == "levenshtein":                
            score  = lv.ratio(x, y) * 100

        # cosin 
        elif self.similarity_algorithm == "cosin":
            score = self.get_cosine(x, y)

        # jaccard
        elif self.similarity_algorithm == "jaccard":
            score = self.jaccard_similarity(x, y)

        else:
            score = -1
            print("Geçersiz Algoritma.")


        return score 

    def text_to_vector(self, text):
        words = WORD.findall(text)
        return Counter(words)


