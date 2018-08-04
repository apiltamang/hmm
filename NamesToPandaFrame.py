import pandas as pd
import numpy as np
import string
import random
import matplotlib.pyplot as plt
import sys
import time
import math
import glob
import re
from tqdm import tqdm

class NamesToPandaFrame:
    def __init__(self, ssa_fn, surname_fn):
        self.ssa_fn = ssa_fn
        self.surname_fn = surname_fn

    def readNames(self):
        
        firstNamesFinal = pd.read_csv(self.ssa_fn)
        lastNamesFinal = pd.read_csv(self.surname_fn)

        # Delete unused columns
        del lastNamesFinal['rank']
        del lastNamesFinal['count']
        del lastNamesFinal['prop100k']
        del lastNamesFinal['cum_prop100k']
        del lastNamesFinal['pctwhite']
        del lastNamesFinal['pctblack']
        del lastNamesFinal['pctapi']
        del lastNamesFinal['pctaian']
        del lastNamesFinal['pct2prace']
        del lastNamesFinal['pcthispanic']

        firstNamesFinal.shape

        lastNamesFinal.shape

        #Remove duplicate names
        firstNamesFinal.drop_duplicates(subset=["Name"], keep='first', inplace=True)
        lastNamesFinal.drop_duplicates(subset=["Name"], keep='first', inplace=True)

        #drop non strings
        lastNamesFinal['keep'] = lastNamesFinal['Name'].apply(lambda x: type(x)==str)
        lastNamesFinal.drop( lastNamesFinal[lastNamesFinal['keep']==False].index.tolist() , inplace=True)

        firstNamesFinal['keep'] = firstNamesFinal['Name'].apply(lambda x: type(x)==str)
        firstNamesFinal.drop( firstNamesFinal[firstNamesFinal['keep']==False].index.tolist() , inplace=True)

        #drop names w/ special chars
        lastNamesFinal['keep'] = lastNamesFinal['Name'].apply(lambda s: len(s)==len(s.encode()))
        lastNamesFinal.drop( lastNamesFinal[lastNamesFinal['keep']==False].index.tolist() , inplace=True)

        firstNamesFinal['keep'] = firstNamesFinal['Name'].apply(lambda s: len(s)==len(s.encode()))
        firstNamesFinal.drop( firstNamesFinal[firstNamesFinal['keep']==False].index.tolist() , inplace=True)

        #set len of names
        lastNamesFinal = lastNamesFinal.assign(Length=lastNamesFinal['Name'].astype('str').str.len())
        lastNamesFinal = lastNamesFinal.reset_index()
        #make sure less than 100
        #print(len(lastNamesFinal['Name'][lastNamesFinal['Length'].idxmax()]))

        firstNamesFinal = firstNamesFinal.assign(Length=firstNamesFinal['Name'].astype('str').str.len())
        firstNamesFinal = firstNamesFinal.reset_index()
        #make sure less than 100
        #print(len(firstNamesFinal['Name'][firstNamesFinal['Length'].idxmax()]))

        allNames = firstNamesFinal.append(lastNamesFinal)

        return allNames
    