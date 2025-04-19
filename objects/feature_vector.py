class FeatureVector(object):

    def __init__(self):
        """
        Initialise the 5 features extracted from each packet.
        """
        self.sport = None
        self.dport = None
        self.time_delta = None
        self.direction = None
        self.flags = None
        self.window = None
        self.payload = None
        self.length = None