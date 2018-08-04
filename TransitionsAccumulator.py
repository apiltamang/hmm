from NamesToPandaFrame import *
import pandas as pd
from tqdm import tqdm
import numpy as np
from StateCharMapper import *

class TransitionsAccumulator:
    
    def __init__(self, ssa_fn='ssa_fn.txt', surname_fn='surnames_2000.csv'):
        ROOT = '/home/apil/Git/fastai/courses/dl1/pp_hmm_explorations'
        self.nLabels = 47
        self.transitions = {}
        self.stateMatrix = np.zeros((self.nLabels, self.nLabels))
        frame = NamesToPandaFrame(f'{ROOT}/{ssa_fn}', f'{ROOT}/{surname_fn}')
        self.dataFrame = frame.readNames()
        self.initialDist = np.zeros((self.nLabels))
        self.specialNeeds = ["a","b","d","e","f","g","h","n","q","r","t"]

        # TODO: As a process of debugging, for now, set transitions between all char -> char, and
        # all digit -> digit be uniform, while transitions between all char <-> digit is zero.
        self.process()
        
        # self.initialDist[:] = 1.0/47
        
        # for i in range(10):
        #    for j in range(10):
        #        self.stateMatrix[i,j] = 1.0/10
        
        # for i in range(10,47):
        #    for j in range(10, 47):
        #        self.stateMatrix[i,j] = 1.0/37
                
    def process(self):
        # accumulate counts of bigrams and letters
        for name in self.dataFrame['Name']:
            self.updateTransitions(name)
            self.updateInitialDist(name)
            
        # convert from dict. to array matrix
        self.convertToStateMatrix()
        
        # finally normalize the probability for state 
        # transitions and initial distributions
        self.normalizeMat()
        self.normalizeDist()
        
        # finally update the transition matrix and distribution of the special need chars.
        # This is simply going to be the values borrowed from their upperCase counterparts
        self.fillMatrixForSpecialNeedsChars()
            
    def genBiGrams(self, word):
        return [word[i:i+2] for i in range(len(word)-1)]

    def updateTransitions(self, word):        
        for bigram in self.genBiGrams(word.upper()):
            if bigram in self.transitions:
                self.transitions[bigram] = self.transitions[bigram]+1
            else:
                self.transitions[bigram] = 1

    def updateInitialDist(self, word):
        for ch in word.upper():
            val = StateCharMapper.charToState(ch)
            self.initialDist[val] += 1
            
    def convertToStateMatrix(self):
        for key, count in tqdm(self.transitions.items()):
            c1 = StateCharMapper.charToState(key[0])
            c2 = StateCharMapper.charToState(key[1])

            self.stateMatrix[c1,c2] = count
    
    def normalizeRow(self, arr):
        tot = sum(arr)    
        if (tot != 0.):
            arr = arr/tot
        return arr
    
    def normalizeMat(self):
        # for classes 0 to 9, make the transitions uniform.
        for i in range(10):
            self.stateMatrix[i,0:10] = self.normalizeRow(np.ones(10))
            
        # for the characters (10 to 36), iterate over stateMatrix rows
        for i in range(10, self.nLabels):
            col = self.stateMatrix[i]
            self.stateMatrix[i] = self.normalizeRow(col)
    
    def normalizeDist(self):
        self.initialDist = self.normalizeRow(self.initialDist)
        
    def prettyPrint(self):
        ''' returns a string of transition matrix. View by copying/pasting on spreadsheet or text processor'''
        
        string = "{:7}".format(" ")
        for k in range(47):
            string += "{:>7}".format(StateCharMapper.stateToChar(k))
        string += "\n"
        for i in range(47):    
            string += "{:>7}".format(StateCharMapper.stateToChar(i))
            for j in range(47):
                string += "{:7.3f}".format(self.stateMatrix[i,j])
            string +="\n"      
        
        return string
    
    def fillMatrixForSpecialNeedsChars(self):
        # update transitions from lower <-> lower characters
        for chFrom in self.specialNeeds:
            
            indxFrom = StateCharMapper.charToState(chFrom)
            indxFromU = StateCharMapper.charToState(chFrom.upper())

            assert self.initialDist[indxFrom] == 0.0
            self.initialDist[indxFrom] = self.initialDist[indxFromU]
            
            for chTo in self.specialNeeds:
                indxTo = StateCharMapper.charToState(chTo)                
                indxToU = StateCharMapper.charToState(chTo.upper())
                
                assert self.stateMatrix[indxFrom, indxTo] == 0.0
                self.stateMatrix[indxFrom, indxTo] = self.stateMatrix[indxFromU, indxToU]
        
        # update transitions from upper -> lower characters
        import string
        for chFrom in string.ascii_uppercase:
            indxFrom = StateCharMapper.charToState(chFrom)
            
            for chTo in self.specialNeeds:
                indxTo = StateCharMapper.charToState(chTo)
                indxToU = StateCharMapper.charToState(chTo.upper())

                assert self.stateMatrix[indxFrom, indxTo] == 0.0
                self.stateMatrix[indxFrom, indxTo] = self.stateMatrix[indxFrom, indxToU]
                
        # update transitions from lower -> upper characters
        for chFrom in self.specialNeeds:            
            indxFrom = StateCharMapper.charToState(chFrom)
            indxFromU = StateCharMapper.charToState(chFrom.upper())
            
            for chTo in string.ascii_uppercase:
                indxTo = StateCharMapper.charToState(chTo)
                
                assert self.stateMatrix[indxFrom, indxTo] == 0.0
                self.stateMatrix[indxFrom, indxTo] = self.stateMatrix[indxFromU, indxTo]
            