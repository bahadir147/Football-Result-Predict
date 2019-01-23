import pickle


def SaveData(data, filename):

    filePath = "Models\\" + filename
    with open(filePath, 'wb') as f:
        pickle.dump(data, f)


def LoadData(filename):
    filePath = "Models\\" + filename
    with open(filePath, 'rb') as f:
        rf = pickle.load(f)
        return rf


