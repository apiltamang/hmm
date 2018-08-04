A HMM layer for the PaperPy project


First, change the ROOT PATH constants for TransitionAccumulator.py and StateCharMapper.py, so that you can load the files: charToClass.p, clasToChar.p, ssa_fn.txt, and surnames_2000.csv in the above classes.

Then, simply instantiate a new Viterbi solver using: 
```
acc = TransitionsAccumulator()
viterbi = Viterbi(acc)
```
and finally call the solver code using:
```
viterbi.solve(prediction_matrix)
```
