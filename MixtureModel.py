from sklearn.mixture import GaussianMixture as GM


class GaussianMixModel(GM):
    def __init__(self):
        super().__init__()
