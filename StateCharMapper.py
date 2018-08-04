import sys
sys.path.append("..")
import string
import pickle

class StateCharMapper:
    '''
    This class takes in any character from 0 - 9, A - Z, and a - z, and maps it into a state
    between 0 to 47. There are certain characters (see `excluded` list) where the lower chars
    are mapped to the states for their upper-case form.
    '''
    
    excluded = ['c', 'i', 'j', 'k', 'l', 'm', 'o', 'p', 's', 'u', 'v', 'w', 'x', 'y', 'z']    
    PRED_ARTIFACTS_ROOT = "/home/apil/Git/fastai/fastai/paperpy"        
    
    # simply a pickled map object defining class -> char relationship 
    '''
    0 --> 0 	1 --> 1 	2 --> 2 	3 --> 3 	4 --> 4 	5 --> 5 	6 --> 6 	7 --> 7 	8 --> 8 	9 --> 9 	
    10 --> A 	11 --> B 	12 --> C 	13 --> D 	14 --> E 	15 --> F 	16 --> G 	17 --> H 	18 --> I 	19 --> J 	
    20 --> K 	21 --> L 	22 --> M 	23 --> N 	24 --> O 	25 --> P 	26 --> Q 	27 --> R 	28 --> S 	29 --> T 	
    30 --> U 	31 --> V 	32 --> W 	33 --> X 	34 --> Y 	35 --> Z 	36 --> a 	37 --> b 	38 --> d 	39 --> e 	
    40 --> f 	41 --> g 	42 --> h 	43 --> n 	44 --> q 	45 --> r 	46 --> t 	
    '''    
    clasToChar = pickle.load(open(f'{PRED_ARTIFACTS_ROOT}/clasToChar.p',"rb"))
    
    # simply a pickled map object defining char -> class relationship    
    '''
    0 --> 0 	1 --> 1 	2 --> 2 	3 --> 3 	4 --> 4 	5 --> 5 	6 --> 6 	7 --> 7 	8 --> 8 	9 --> 9 	
    A --> 10 	B --> 11 	C --> 12 	D --> 13 	E --> 14 	F --> 15 	G --> 16 	H --> 17 	I --> 18 	J --> 19 	
    K --> 20 	L --> 21 	M --> 22 	N --> 23 	O --> 24 	P --> 25 	Q --> 26 	R --> 27 	S --> 28 	T --> 29 	
    U --> 30 	V --> 31 	W --> 32 	X --> 33 	Y --> 34 	Z --> 35 	a --> 36 	b --> 37 	d --> 38 	e --> 39 	
    f --> 40 	g --> 41 	h --> 42 	n --> 43 	q --> 44 	r --> 45 	t --> 46 	
    '''    
    charToClas = pickle.load(open(f"{PRED_ARTIFACTS_ROOT}/charToClas.p","rb"))
    
    @classmethod
    def charToState(cls,ch):
        ch_mod = ch

        if ch in cls.excluded:
            ch_mod = ch.upper()
        return cls.charToClas[ch_mod]
    
    @classmethod    
    def stateToChar(cls,val):
        return cls.clasToChar[val]
