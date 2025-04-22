import torch.nn as nn

# LSTM Autoencoder
class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len, input_dim, hidden_dim, latent_dim):
        super(LSTMAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.hidden_to_latent = nn.Linear(hidden_dim, latent_dim)
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        _, (h_n, _) = self.encoder(x)
        latent = self.hidden_to_latent(h_n[-1])
        hidden = self.latent_to_hidden(latent).unsqueeze(1).repeat(1, self.seq_len, 1)
        decoded, _ = self.decoder(hidden)
        out = self.output_layer(decoded)
        return out, latent


# Stacked Autoencoder
class StackedAutoencoder(nn.Module):
    def __init__(self, seq_len, input_dim=8, hidden_dims=[32, 16, 8], latent_dims=[16, 8, 4]):
        super(StackedAutoencoder, self).__init__()
        assert len(hidden_dims) == len(latent_dims)
        self.autoencoders = nn.ModuleList()
        self.input_dims = [input_dim] + latent_dims[:-1]

        for i in range(len(hidden_dims)):
            ae = LSTMAutoencoder(seq_len, self.input_dims[i], hidden_dims[i], latent_dims[i])
            self.autoencoders.append(ae)

    def forward(self, x):
        code = x
        for i, ae in enumerate(self.autoencoders):
            _, code = ae(code)
            if i + 1 < len(self.autoencoders):
                code = code.unsqueeze(1).repeat(1, ae.seq_len, 1)
        return self.decode(code)

    def decode(self, code):
        x = code
        for ae in reversed(self.autoencoders):
            hidden = ae.latent_to_hidden(x).unsqueeze(1).repeat(1, ae.seq_len, 1)
            decoded, _ = ae.decoder(hidden)
            x = ae.output_layer(decoded)
        return x

    def encode(self, x):
        code = x
        for i, ae in enumerate(self.autoencoders):
            _, code = ae(code)
            if i + 1 < len(self.autoencoders):
                code = code.unsqueeze(1).repeat(1, ae.seq_len, 1)
        return code
