class Arguments:
    def __init__(self, d):
        self.as_dictionary = d
        for key, value in self.as_dictionary.items():
            setattr(self, key, value)
        
def get_args(d):
    return Arguments(d)