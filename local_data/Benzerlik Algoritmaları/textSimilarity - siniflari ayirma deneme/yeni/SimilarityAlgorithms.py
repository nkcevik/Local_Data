import pandas as pd 
import numpy as np
import Levenshtein as lv
import re, math
from collections import Counter
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from CleanData import CleanData


class SimilarityAlgorithms:

    def _reset_cache(self):
        self.score = None

    def __init__(self, str1, str2, similarity_algorithm):
        self.x, self.y = str1, str2
        self.similarity_algorithm = similarity_algorithm
        self._reset_cache()
    
    def jaccard_similarity(self):
        vec1 = self.text_to_vector(self.x)
        vec2 = self.text_to_vector(self.y)

        intersection_cardinality = len(set(vec1).intersection(set(vec2)))
        union_cardinality = len(set(vec1).union(set(vec2)))
        jaccard = intersection_cardinality / float(union_cardinality)

        return jaccard * 100

    def get_cosine(self):
        vec1 = self.text_to_vector(self.x)
        vec2 = self.text_to_vector(self.y)

        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])

        sum1 = sum([vec1[x]**2 for x in vec1.keys()])
        sum2 = sum([vec2[x]**2 for x in vec2.keys()])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)

        if not denominator:
            return 0.0 
        else:
            return (float(numerator) / denominator) * 100
            
    def scores(self):
        
        # levenshtein    
        if self.similarity_algorithm == "levenshtein":                
            score  = lv.ratio() * 100

        # cosin 
        elif self.similarity_algorithm == "cosin":
            score = self.get_cosine()

        # jaccard
        elif self.similarity_algorithm == "jaccard":
            score = self.jaccard_similarity()

        else:
            score = -1
            print("Geçersiz Algoritma.")


        return score 

    def text_to_vector(self, text):
        WORD = re.compile(r'\w+')
        words = WORD.findall(text)
        return Counter(words)