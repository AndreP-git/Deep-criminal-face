import os

class DirNames():
    def __init__(self, path) -> None:
        self.path = path
        self.test_names = []
        self.train_names = []
        self.all_names = []

    def search(self):
        for name in os.listdir(self.path + 'train'):
            name = name.replace('_', ' ').lower()
            self.train_names.append(name)
        
        for name in os.listdir(self.path + 'test'):
            name = name.replace('_', ' ').lower()
            self.train_names.append(name)

        self.all_names = self.train_names + self.test_names

