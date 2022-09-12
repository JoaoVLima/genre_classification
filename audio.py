

class Audio:
    def __init__(self, *args, **kwargs):
        self.data = kwargs.get('data')
        self.sampling_rate = kwargs.get('sampling_rate')