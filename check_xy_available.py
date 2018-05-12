import json
import numpy as np

filename='temp/x.txt'
data=json.load(open(filename))
data=np.array(data)
print(np.isnan(data).sum())