class Arguments:
    def __init__(self, d):
        for key, value in d.items():
            setattr(self, key, value)

def get_args(d):
    return Arguments(d)