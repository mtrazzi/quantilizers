class RunningMean:
    def __init__(self):
        self.total = 0
        self.length = 0
    def new(self, element):
        self.total += element
        self.length += 1
        self.mean = self.total/self.length
        return self.mean
    __call__ = new