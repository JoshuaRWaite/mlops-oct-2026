"""The autoencoder. Training and serving both import it from here."""
import torch.nn as nn


class AutoEncoder(nn.Module):
    """1x128x128 grayscale image -> 8x8x8 latent (512 numbers) -> 1x128x128 image.

    The latent is 32x smaller than the input (512 vs 16,384 values), so the network
    can't just copy pixels. It has to learn what matters about a house photo.
    """

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),    # 16 x 64 x 64
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),   # 32 x 32 x 32
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),   # 64 x 16 x 16
            nn.Conv2d(64, 8, 3, stride=2, padding=1),               #  8 x  8 x  8  <- bottleneck
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 64, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),   # 64 x 16 x 16
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),  # 32 x 32 x 32
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),  # 16 x 64 x 64
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),              #  1 x 128 x 128
            nn.Sigmoid(),  # pixel values back in [0, 1]
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))
