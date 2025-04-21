class FeatureVector:

    def __init__(self):
        """
        Initialise the 5 features extracted from each packet.
        """
        self.time_delta = None
        self.direction = None
        self.flags = None
        self.window = None
        self.length = None
        self.ttl = None