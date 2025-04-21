from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import jensenshannon


class DeviceClassifier:
    def __init__(self, n_clusters=None, alpha=1, threshold=0.3, max_k=10):
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.threshold = threshold
        self.kmeans = None
        self.label_posteriors = {}
        self.max_k = max_k

    def _dirichlet_mean(self, cluster_ids):
        counts = np.bincount(cluster_ids, minlength=self.n_clusters)
        alpha_post = counts + self.alpha
        return alpha_post / alpha_post.sum()

    def _select_best_k(self, X):
        best_score = -1
        best_k = 2
        best_model = None

        for k in range(2, self.max_k + 1):
            model = KMeans(n_clusters=k, random_state=42)
            labels = model.fit_predict(X)

            try:
                score = silhouette_score(X, labels)
            except:
                continue  # In case a cluster ends up empty

            if score > best_score:
                best_score = score
                best_k = k
                best_model = model

        self.kmeans = best_model
        self.n_clusters = best_k
        print(f"Selected optimal k={best_k} with silhouette score={best_score:.3f}")

    def fit(self, X, y):
        if self.n_clusters is None:
            self._select_best_k(X)
        else:
            self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42).fit(X)

        cluster_ids = self.kmeans.predict(X)

        label_to_clusters = defaultdict(list)
        for cid, label in zip(cluster_ids, y):
            label_to_clusters[label[0]].append(cid)

        for label, cids in label_to_clusters.items():
            self.label_posteriors[label] = self._dirichlet_mean(np.array(cids))

    
    def predict(self, X_test):
        """
        X_test: np.ndarray of shape (n_samples, n_features)
        Returns: list of predicted labels, one for each row in X_test
        """
        predictions = []
        probs = []
        cluster_ids = self.kmeans.predict(X_test)

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

            predictions.append(best_label if best_score < self.threshold else "Unknown")
            probs.append(best_score)

        return predictions, probs
