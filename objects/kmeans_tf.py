import tensorflow as tf
import numpy as np
from collections import Counter

class KMeansTF:
    def __init__(self, n_clusters=3, n_iterations=150, seed=None):
        self.n_clusters = n_clusters
        self.n_iterations = n_iterations
        self.centroids = None
        self.labels_ = None
        self.cluster_to_label = {}
        self.seed = seed

    def fit(self, X, y_true=None):
        # Ensure 2D input
        if len(X.shape) == 3:
            X = tf.reduce_mean(X, axis=1)
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        n_samples = tf.shape(X)[0]

        # Initialize centroids
        if self.seed is not None:
            tf.random.set_seed(self.seed)
        indices = tf.random.shuffle(tf.range(n_samples))[:self.n_clusters]
        self.centroids = tf.gather(X, indices)

        for _ in range(self.n_iterations):
            # Compute distances and assign clusters
            distances = tf.reduce_sum(tf.square(tf.expand_dims(X, 1) - tf.expand_dims(self.centroids, 0)), axis=2)
            cluster_ids = tf.argmin(distances, axis=1)

            # Update centroids
            new_centroids = []
            for c in range(self.n_clusters):
                mask = tf.equal(cluster_ids, c)
                points = tf.boolean_mask(X, mask)
                if tf.shape(points)[0] > 0:
                    new_centroids.append(tf.reduce_mean(points, axis=0))
                else:
                    # Random fallback if cluster is empty
                    new_centroids.append(tf.gather(X, tf.random.uniform([], maxval=n_samples, dtype=tf.int32)))
            self.centroids = tf.stack(new_centroids)

        self.labels_ = cluster_ids.numpy()

        if y_true is not None:
            self._assign_cluster_labels(np.array(y_true))

    def _assign_cluster_labels(self, y_true):
        self.cluster_to_label = {}
        for c in range(self.n_clusters):
            mask = self.labels_ == c
            labels = [str(l.item()) if isinstance(l, np.ndarray) else str(l) for l in y_true[mask]]
            if np.sum(mask) > 0:
                self.cluster_to_label[c] = Counter(labels).most_common(1)[0][0]
            else:
                self.cluster_to_label[c] = -1

    def predict(self, X, if_unknown = False):
        if len(X.shape) == 3:
            X = tf.reduce_mean(X, axis=1)
        X = tf.convert_to_tensor(X, dtype=tf.float32)

        distances = tf.reduce_sum(tf.square(tf.expand_dims(X, 1) - tf.expand_dims(self.centroids, 0)), axis=2)
        cluster_ids = tf.argmin(distances, axis=1).numpy()

        # Soft assignment: higher distance → lower probability
        neg_distances = -distances
        probs = tf.nn.softmax(neg_distances, axis=1).numpy()  # Shape: (n_samples, n_clusters)

        
        # Map clusters to original labels if available
        if self.cluster_to_label:
            mapped_labels = np.vectorize(self.cluster_to_label.get)(cluster_ids)
        else:
            mapped_labels = cluster_ids
        
        # Return the label and its associated cluster probability
        max_probs = np.max(probs, axis=1)

        if if_unknown:
            for i in range(len(mapped_labels)):
                if max_probs[i] < 0.45:
                    mapped_labels[i] = "Unknown"

        
        return mapped_labels, max_probs.tolist()