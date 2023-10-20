import json
import os

def json_load(name, path, verbose=False):    
    full_name= (os.path.join(path,name+ '.json'))
    with open(full_name, 'r') as ifile:
        data=json.load(ifile)
    if verbose:
        print('loaded from: ',full_name)
    return data