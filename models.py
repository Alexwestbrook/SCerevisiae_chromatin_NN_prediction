import torch
from torch import nn


class MnaseEtienneNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_stack = nn.Sequential(
            nn.Conv1d(4, 64, 3, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Conv1d(64, 16, 8, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(16),
            nn.Dropout(0.2),
            nn.Conv1d(16, 80, 8, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(80),
            nn.Dropout(0.2),
            nn.Flatten(),
            nn.Linear(80 * 250, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        return self.conv_stack(x)


class ResidualConcatLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        return torch.cat([x, self.layer(x)], dim=1)


class PooledConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_size):
        super().__init__()
        self.pooled_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(pool_size),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        return self.pooled_conv(x)


class DilatedConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.dilated_conv = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding="same",
                dilation=dilation,
            ),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        return self.dilated_conv(x)


class BassenjiEtienneNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stack = nn.Sequential(
            self.pooled_conv_wrapper(4, 32, 12, pool_size=8),
            self.pooled_conv_wrapper(32, 32, 5, pool_size=4),
            self.pooled_conv_wrapper(32, 32, 5, pool_size=4),
            self.dilated_conv_wrapper(32, 16, 5, dilation=2),
            ResidualConcatLayer(self.dilated_conv_wrapper(16, 16, 5, dilation=4)),
            ResidualConcatLayer(self.dilated_conv_wrapper(32, 16, 5, dilation=8)),
            ResidualConcatLayer(self.dilated_conv_wrapper(48, 16, 5, dilation=16)),
            nn.Conv1d(64, 1, 1),
            nn.ReLU(),
            nn.Flatten(),
        )

    def pooled_conv_wrapper(self, in_channels, out_channels, kernel_size, pool_size):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(pool_size),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(0.2),
        )

    def dilated_conv_wrapper(self, in_channels, out_channels, kernel_size, dilation):
        return nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding="same",
                dilation=dilation,
            ),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        return self.conv_stack(x)


class BassenjiMnaseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stack = nn.Sequential(
            self.pooled_conv_wrapper(4, 32, 12, pool_size=4),
            self.pooled_conv_wrapper(32, 32, 5, pool_size=2),
            self.pooled_conv_wrapper(32, 32, 5, pool_size=2),
            self.dilated_conv_wrapper(32, 16, 5, dilation=2),
            ResidualConcatLayer(self.dilated_conv_wrapper(16, 16, 5, dilation=4)),
            ResidualConcatLayer(self.dilated_conv_wrapper(32, 16, 5, dilation=8)),
            ResidualConcatLayer(self.dilated_conv_wrapper(48, 16, 5, dilation=16)),
            nn.Conv1d(64, 1, 1),
            nn.ReLU(),
            nn.Flatten(),
        )

    def pooled_conv_wrapper(self, in_channels, out_channels, kernel_size, pool_size):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(pool_size),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(0.2),
        )

    def dilated_conv_wrapper(self, in_channels, out_channels, kernel_size, dilation):
        return nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding="same",
                dilation=dilation,
            ),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        return self.conv_stack(x)


class BassenjiMultiNetwork(nn.Module):
    def __init__(self, n_tracks=2):
        super().__init__()
        self.conv_stack = nn.Sequential(
            PooledConvLayer(4, 32, 12, pool_size=4),
            PooledConvLayer(32, 32, 5, pool_size=2),
            PooledConvLayer(32, 32, 5, pool_size=2),
            DilatedConvLayer(32, 16, 5, dilation=2),
            ResidualConcatLayer(DilatedConvLayer(16, 16, 5, dilation=4)),
            ResidualConcatLayer(DilatedConvLayer(32, 16, 5, dilation=8)),
            ResidualConcatLayer(DilatedConvLayer(48, 16, 5, dilation=16)),
            nn.Conv1d(64, n_tracks, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.conv_stack(x)
        return torch.transpose(x, 1, 2)


ARCHITECTURES = {
    "BassenjiMnaseNetwork": BassenjiMnaseNetwork,
    "BassenjiEtienneNetwork": BassenjiEtienneNetwork,
    "BassenjiMultiNetwork": BassenjiMultiNetwork,
}
