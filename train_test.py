# import numpy as np
# import torch
# import torch.nn as nn

# from collections import defaultdict, Counter
# from scipy.stats import dirichlet
# from scipy.spatial.distance import jensenshannon
from sklearn.cluster import DBSCAN
# from sklearn.metrics import silhouette_score

import tensorflow as tf
from objects.lstm_tf import StackedLSTMAutoencoder
# from objects.lstm import StackedAutoencoder
# from util import convert_to_tensor


def train_lstm_ae(x_train):
    """
    """
    tf.experimental.numpy.experimental_enable_numpy_behavior()

    # Convert the dataframes to tensors
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)

    print(x_train.shape)

    n_seq, seq_len, n_features = x_train.shape
    
    # ---------------------------
    # Instantiate model
    # ---------------------------
    epochs = 20
    lr = 1e-3
    model = StackedLSTMAutoencoder(seq_len, n_features)
    model.fit(x_train, epochs)
    
    # model.train()
    # for i, ae in enumerate(model.autoencoders):
    #     print(f"\nTraining Autoencoder Layer {i+1}/{len(model.autoencoders)}")
    #     optimizer = torch.optim.Adam(ae.parameters(), lr=lr)
    #     criterion = nn.MSELoss()

    #     x_input = x_train
    #     # Get input to this layer
    #     if i > 0:
    #         with torch.no_grad():
    #             for j in range(i):
    #                 _, code = model.autoencoders[j](x_input)
    #                 code = code.unsqueeze(1).repeat(1, model.autoencoders[j].seq_len, 1)
    #                 x_input = code
    
    #     for epoch in range(epochs):
    #         ae.train()
    #         optimizer.zero_grad()
    #         output, _ = ae(x_input)
    #         loss = criterion(output, x_input)
    #         loss.backward()
    #         optimizer.step()
    #         if epoch % 5 == 0:
    #             print(f"  Epoch {epoch} - Loss: {loss.item():.4f}")
    #     print(f"  Epoch {epoch} - Loss: {loss.item():.4f}")
    
    return model


def encode_data(model, data):
    # Convert the dataframes to tensors
    data = tf.convert_to_tensor(data, dtype=tf.float32)
    encoded_data = model.encode(data) 

    return encoded_data
