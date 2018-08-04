import operator
import numpy as np
from StateCharMapper import *
import sys; sys.path.append("..")

class ViterbiSolver:
    
    def __init__(self, acc):
        '''
        acc: an instance of TransitionsAccumulator (see type provided with this package)        
        '''
        self.acc = acc
        self.priors = acc.initialDist     # initial distribution across the 47 states
        self.transition = acc.stateMatrix # transition probabilities across a 47 x 47 states
    
    def solve(self, emission):
        """Return the best path, given an HMM model and a sequence of observations
        emission: array(nStates, nObservations). Example, we have 47 states in this
                  char recognition problem. Then, if an observation for the 
                  word 'Apil' is passed to this method, then the emisison matrix is
                  as (47, 4), where each column is a probability vector for the given
                  word.
                  
                  A     P    I    L
                  x1    x2   x3   x4   
                  y1    y2   y3   y4
                   .     .    .    . 
                   .     .    .    .
                   .     .    .    .
                   .     .    .    .
                   .     .    .    .
                   
                  z1     z2   z3  z4   
                  
                  where each vector (x1, y1, ... z1) (x2, y2, ...., z2) etc represents a 
                  prediction probability across each class in the 47 states we have.
        """
        
        # A - initialise stuff
        nSamples = len(emission[0])
        nStates = self.transition.shape[0] # number of states
        c = np.zeros(nSamples) #scale factors (necessary to prevent underflow)
        viterbi = np.zeros((nStates,nSamples)) # initialise viterbi table
        psi = np.zeros((nStates,nSamples)) # initialise the best path table
        best_path = np.zeros(nSamples).astype(np.int32); # this will be your output

        # B- appoint initial values for viterbi and best path (bp) tables - Eq (32a-32b)
        viterbi[:,0] = self.priors.T * emission[:,0]
        c[0] = 1.0/np.sum(viterbi[:,0])
        viterbi[:,0] = c[0] * viterbi[:,0] # apply the scaling factor

        # C- Do the iterations for viterbi and psi for time>0 until T
        for t in range(1,nSamples): # loop through time
            for s in range (0,nStates): # loop through the states @(t-1)
                trans_p = viterbi[:,t-1] * self.transition[:,s]
                psi[s,t], viterbi[s,t] = max(enumerate(trans_p), key=operator.itemgetter(1))
                viterbi[s,t] = viterbi[s,t] * emission[s,t]

            c[t] = 1.0/np.sum(viterbi[:,t]) # scaling factor
            viterbi[:,t] = c[t] * viterbi[:,t]

        # D - Back-tracking
        best_path[nSamples-1] =  viterbi[:,nSamples-1].argmax() # last state
        for t in range(nSamples-1,0,-1): # states of (last-1)th to 0th time step
            best_path[t-1] = psi[best_path[t],t]

        return best_path
    