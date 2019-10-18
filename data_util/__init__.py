import os

class Path:
    def __init__(self, path):
        self.path = path
        if not os.path.exists(path):
            os.mkdir(path)
    def __call__(self, *names):
        return os.path.join(*((self.path, ) + names))