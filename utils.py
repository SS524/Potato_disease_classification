import pickle

def load_obj(file_path):
    with open(file_path,'rb') as load_obj:
        return pickle.load(load_obj)