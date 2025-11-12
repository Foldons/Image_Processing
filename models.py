import torch
import torch.nn as nn


class Autoencoder_CC_old(nn.Module):
    def __init__(self, in_channels=1, out_channels=16, latent_dim=64, input_shape=(1,28,28)):
        super().__init__()

        # ----- Encoder -----
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, 2*out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(2*out_channels, 4*out_channels, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
        )

        # Compute flattened size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            conv_out = self.encoder_conv(dummy)
            self.flattened_size = conv_out.numel()
            self.conv_shape = conv_out.shape[1:]  # (C,H,W)

        # Linear layers to latent space
        self.fc_enc = nn.Linear(self.flattened_size, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.flattened_size)

        # ----- Decoder -----
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(4*out_channels, 2*out_channels, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(2*out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(out_channels, in_channels, kernel_size=3, stride=1, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)  # flatten
        z = self.fc_enc(x)
        return z

    def decode(self, z):
        x = self.fc_dec(z)
        x = x.view(x.size(0), *self.conv_shape)  # reshape to conv feature map
        x = self.decoder_conv(x)
        return x

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z

class Autoencoder_CC(nn.Module):
    def __init__(self, in_channels=1, out_channels=16, latent_dim=64, input_shape=(1, 28, 28)):
        super().__init__()

        # ----- Encoder -----
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),  # 28 → 14
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),  # 14 → 7
            nn.ReLU(),
            nn.Conv2d(out_channels, 2 * out_channels, kernel_size=3, stride=2, padding=1),  # 7 → 4
            nn.ReLU(),
            nn.Conv2d(2 * out_channels, 4 * out_channels, kernel_size=5, stride=2, padding=1),  # 4 → 2
            nn.ReLU(),
        )

        # compute conv output size
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            conv_out = self.encoder_conv(dummy)
            self.flattened_size = conv_out.numel()
            self.conv_shape = conv_out.shape[1:]  # (C, H, W)

        # bottleneck FC
        self.fc_enc = nn.Linear(self.flattened_size, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.flattened_size)

        # ----- Decoder (fixed!) -----
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(4 * out_channels, 2 * out_channels, kernel_size=5, stride=2, padding=1, output_padding=1),  # 2 → 4
            nn.ReLU(),
            nn.ConvTranspose2d(2 * out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),  # 4 → 8
            nn.ReLU(),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=0),  # 8 → 15
            nn.ReLU(),
            nn.ConvTranspose2d(out_channels, in_channels, kernel_size=4, stride=2, padding=2),  # 15 → 30
            nn.Sigmoid()
        )


    def encode(self, x):
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)
        return self.fc_enc(x)

    def decode(self, z):
        x = self.fc_dec(z)
        x = x.view(x.size(0), *self.conv_shape)  # reshape to (C,H,W)
        x = self.decoder_conv(x)
        return x

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z

class Autoencoder_CC_cifar_working(nn.Module):
    def __init__(self, in_channels=3, out_channels=16, latent_dim=64, input_shape=(3, 32, 32)):
        super().__init__()

        # ----- Encoder -----
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),  # 28 → 14
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),  # 14 → 7
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(out_channels, 2 * out_channels, kernel_size=3, stride=2, padding=1),  # 7 → 4
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(2 * out_channels, 4 * out_channels, kernel_size=5, stride=2, padding=1),  # 4 → 2
            nn.ReLU(),
        )

        # compute conv output size
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            conv_out = self.encoder_conv(dummy)
            self.flattened_size = conv_out.numel()
            self.conv_shape = conv_out.shape[1:]  # (C, H, W)

        # bottleneck FC
        self.fc_enc = nn.Linear(self.flattened_size, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.flattened_size)

        # ----- Decoder (fixed!) -----
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(4 * out_channels, 2 * out_channels, kernel_size=5, stride=2, padding=1, output_padding=1),  # 2 → 4
            nn.ReLU(),
            nn.ConvTranspose2d(2 * out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),  # 4 → 8
            nn.ReLU(),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),  # 8 → 15
            nn.ReLU(),
            nn.ConvTranspose2d(out_channels, in_channels, kernel_size=4, stride=2, padding=1),  # 15 → 30
            nn.Sigmoid()
        )


    def encode(self, x):
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)
        #x = nn.functional.normalize(x) # kill this if not working
        return self.fc_enc(x)

    def decode(self, z):
        x = self.fc_dec(z)
        x = x.view(x.size(0), *self.conv_shape)  # reshape to (C,H,W)
        x = self.decoder_conv(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        z = self.fc_enc(x)
        recon = self.decode(z)
        return recon, z


class Autoencoder_CC_cifar(nn.Module):
    def __init__(self, latent_dim , input_shape=(3, 32, 32)):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 8, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        dummy = torch.zeros(1,3, 32,32)
        enc_out = self.encoder(dummy)
        self.flatten_dim = enc_out.view(1,-1).size(1)

        # bottleneck FC
        self.fc_enc = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.flatten_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(8, 8, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x,1)
        z = self.fc_enc(x)

        x = self.fc_dec(z)
        x = x.view(x.size(0),8, 32//8 , 32//8)
        x = self.decoder(x)
        return x , z


class Autoencoder_CNN_RGB(nn.Module):
    def __init__(self, in_channels=3, out_channels=32, latent_dim=64, input_shape=(1,256,256)):
        super().__init__()

        # ----- Encoder -----
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels*2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels*2, 4*out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(4*out_channels, 8*out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8*out_channels, 8*out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        # Compute flattened size dynamically
        with torch.no_grad():
            dummy = torch.randn(1,3,256,256)
            conv_out = self.encoder_conv(dummy)
            self.flattened_size = conv_out.numel()
            self.conv_shape = conv_out.shape[1:]  # (C,H,W)

        # Linear layers to latent space
        self.fc_enc = nn.Linear(self.flattened_size, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.flattened_size)

        # ----- Decoder -----
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(8*out_channels, 8*out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(8*out_channels, 4*out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(4*out_channels, 2*out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(2*out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(out_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)  # flatten
        z = self.fc_enc(x)
        return z

    def decode(self, z):
        x = self.fc_dec(z)
        x = x.view(x.size(0), *self.conv_shape)  # reshape to conv feature map
        x = self.decoder_conv(x)
        return x

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


class Autoencoder_CNN_RGB_claude(nn.Module):
    def __init__(self, in_channels=3, base_channels=32, latent_dim=512):
        super().__init__()

        # Encoder: 256->128->64->32->16
        self.encoder = nn.Sequential(
            # Block 1: 256x256 -> 128x128
            nn.Conv2d(in_channels, base_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),

            # Block 2: 128x128 -> 64x64
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),

            # Block 3: 64x64 -> 32x32
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),

            # Block 4: 32x32 -> 16x16
            nn.Conv2d(base_channels * 4, base_channels * 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),

            # Block 5: 16x16 -> 8x8
            nn.Conv2d(base_channels * 8, base_channels * 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
        )

        # Latent bottleneck
        self.flatten_size = base_channels * 8 * 8 * 8  # 256*8*8
        self.fc_enc = nn.Linear(self.flatten_size, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.flatten_size)

        # Decoder: 8->16->32->64->128->256
        self.decoder = nn.Sequential(
            # Block 1: 8x8 -> 16x16
            nn.ConvTranspose2d(base_channels * 8, base_channels * 8, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),

            # Block 2: 16x16 -> 32x32
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),

            # Block 3: 32x32 -> 64x64
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),

            # Block 4: 64x64 -> 128x128
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),

            # Block 5: 128x128 -> 256x256
            nn.ConvTranspose2d(base_channels, in_channels, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.fc_enc(x)

    def decode(self, z):
        x = self.fc_dec(z)
        x = x.view(x.size(0), 256, 8, 8)
        return self.decoder(x)

    def forward(self, x):
        z = self.encode(x)
        x = self.decode(z)
        return x,z


class Autoencoder_linear(nn.Module):
    def __init__(self, latent_dim=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim,784),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed, encoded



if "__main__" == __name__:
    num_epochs = 100
    dataloader = torch.load("dataset.pt")
    batch_maps = dataloader['occupancy_maps']

    autoencoder = MapAutoencoder(latent_dim=128)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-4)
    criterion = nn.MSELoss()  # map reconstruction loss

    # Example training loop
    for epoch in range(num_epochs):
        epoch_loss = 0
        for i in range(0, len(batch_maps), 32):
            x = batch_maps[i:i + 32]
            recon, _ = autoencoder(x)
            loss = criterion(recon, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss :.6f}")

