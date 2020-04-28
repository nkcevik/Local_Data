import pandas as pd 
import numpy as np

class CleanData:
 
    def __init__(self, df, unique_column, column):
        self.df = df
        self.unique_column = unique_column
        self.column = column

        return 

    def clean_dataframe(self):
        grup_column, score_column = "GRUP", "SCORE"
        self.df[grup_column] = pd.Series(np.zeros(len(self.df)))
        self.df[score_column] = pd.Series(np.zeros(len(self.df)))

        self.df = self.df[[self.unique_column, self.column, grup_column, score_column]]        

        list_of_deleted_index = self.df[self.df[self.column] == " "].index
        list_of_deleted_index = list_of_deleted_index.append(self.df[self.df[self.column].str.contains('IPTAL', na=True)].index)
        
        # Lenght of deleted index 
        # print(len(list_of_deleted_index))
        
        for index in list_of_deleted_index:
            df = df.drop(index)

        self.df[self.column] = self.df[self.column].replace({'=':''}, regex=True)
        
        return df    

    def all_clean_dataframe(self):
        grup_lev, grup_cos, grup_jac = "GRUP_LEV", "GRUP_COS", "GRUP_JAC"
        score_lev, score_cos, score_jac = "SCORE_LEV", "SCORE_COS", "SCORE_JAC"

        self.df[grup_lev], self.df[score_lev], self.df[grup_cos], self.df[score_cos], self.df[grup_jac], self.df[score_jac] = pd.Series(np.zeros(len(self.df)))
        self.df = self.df[[self.unique_column, self.column, grup_lev, score_lev, grup_cos, score_cos, grup_jac, score_jac]]        

        list_of_deleted_index = self.df[self.df[self.column] == " "].index
        list_of_deleted_index = list_of_deleted_index.append(self.df[self.df[self.column].str.contains('IPTAL', na=True)].index)
        
        # Lenght of deleted index 
        # print(len(list_of_deleted_index))
        
        for index in list_of_deleted_index:
            self.df = self.df.drop(index)

        self.df[self.column] = self.df[self.column].replace({'=':''}, regex=True)
        
        return self.df 
