from sklearn.decomposition import PCA
import numpy as np

class PCACompressor:
    def __init__(self):
        self.pca = PCA()

    def fit(self, data, thresh=0.999):
        self.pca.fit(data)

        sums = 1 - self.pca.explained_variance_
        dims = np.sum(sums < thresh)

        self.pca = PCA(n_components=dims)
        self.pca.fit(data)

    def transform(self, data):
        return self.pca.transform(data)
        
    def fit_transform(self, data, thresh=0.999):
        self.fit(data, thresh)
        return self.transform(data)

