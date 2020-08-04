from torch import nn


class Compressor(nn.Module):
    def __init__(
        self, embed_dim, hid_size, dropout, n_classes, n_layers, bidirectional
    ):
        super().__init__()
        self.table = nn.Embedding(n_classes, embed_dim)
        self.rnn = nn.GRU(
            input_size=embed_dim,
            hidden_size=hid_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.bit_predictor = nn.Sequential(
            nn.Linear(hid_size * (2 if bidirectional else 1), hid_size),
            nn.SELU(),
            nn.Linear(hid_size, n_classes),
        )

    def forward(self, x):
        x = self.table(x)  # BL -> BLE
        x, _ = self.rnn(x)  # BLE -> BL(2H)
        x = self.bit_predictor(x)  # BL(2H) -> BLH -> BL1
        return x
