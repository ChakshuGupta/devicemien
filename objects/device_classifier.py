from sklearn.metrics import silhouette_score
# from sklearn.cluster import KMeans
from kmeans_pytorch import kmeans, kmeans_predict
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import jensenshannon
import torch


class DeviceClassifier:
    def __init__(self, n_clusters=None, unique_label=2, alpha=0.1, threshold=0.5, max_k=10):
        self.n_clusters = n_clusters
        self.unique_label = unique_label
        self.alpha = alpha
        self.threshold = threshold
        self.kmeans = None
        self.label_posteriors = {}
        self.max_k = max_k
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

    def _dirichlet_mean(self, cluster_ids):
        counts = np.bincount(cluster_ids, minlength=self.n_clusters)
        alpha_post = counts + self.alpha
        return alpha_post / alpha_post.sum()

    def _select_best_k(self, X):
        best_score = -1
        best_k = self.unique_label
        best_model = None
        self.labels = None

        for k in range(2, self.max_k + 1):
            # model = KMeans(n_clusters=k, tolerance=1e-4, distance='euclidean')
            labels, cluster_centers = kmeans(X=X, num_clusters=k, distance='euclidean', device=self.device, seed=1234)

            try:
                score = silhouette_score(X, labels, metric='euclidean')
            except:
                continue  # In case a cluster ends up empty

            if score > best_score:
                best_score = score
                best_k = k
                best_model = cluster_centers
                self.labels = labels

        self.kmeans = best_model
        self.n_clusters = best_k
        print(f"Selected optimal k={best_k} with silhouette score={best_score:.3f}")

    def fit(self, X, y):
        if self.n_clusters is None:
            self._select_best_k(X)
        else:
            self.labels, self.kmeans = kmeans(X=X, num_clusters=self.n_clusters, distance='euclidean', device=self.device, seed=1234)

        cluster_ids = kmeans_predict(X, self.kmeans, 'euclidean', device=self.device)

        label_to_clusters = defaultdict(list)
        for cid, label in zip(cluster_ids, y):
            if type(label) is np.ndarray:
                label = label[0]
            label_to_clusters[label].append(cid)
    
        for label, cids in label_to_clusters.items():
            self.label_posteriors[label] = self._dirichlet_mean(np.array(cids))
        print(self.label_posteriors)

    
    def predict(self, X_test, if_unknown = False):
        """
        X_test: np.ndarray of shape (n_samples, n_features)
        Returns: list of predicted labels, one for each row in X_test
        """
        predictions = []
        probs = []
        cluster_ids = kmeans_predict(X_test, self.kmeans, 'euclidean', device=self.device)

        # For each row (flow), make a prediction
        for i in range(len(X_test)):
            flow_cid = cluster_ids[i]
            flow_dist = self._dirichlet_mean([flow_cid])  # just 1 flow

            best_label = None
            best_score = float("inf")

            for label, known_dist in self.label_posteriors.items():
                js = jensenshannon(flow_dist, known_dist)
                if js < best_score:
                    best_score = js
                    best_label = label

            if if_unknown:
                predictions.append(best_label if best_score < self.threshold else "Unknown")
            else:
                predictions.append(best_label)
            probs.append(best_score)

        return predictions, probs
