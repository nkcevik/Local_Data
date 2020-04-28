import pandas as pd 
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# import similarity as sm
from Similarity import Similarity

# Benzerlik tanımları: https://towardsdatascience.com/overview-of-text-similarity-metrics-3397c4601f50
# https://medium.com/@adriensieg/text-similarities-da019229c894

if __name__ == '__main__':

    path = os.path.dirname(__file__) + "/ornek_malzeme_verisi_active.xlsx"
    dataset = pd.read_excel(path)

    # levenshtein
    # jaccard
    # cosin

    # example1
    ts = Similarity("levenshtein", 80, "MLZ_KISA_TANIM_TR", "MLZNO", dataset)
    # print(ts.column)
    # print(ts.similarity_algorithm)
    ts.fit()




