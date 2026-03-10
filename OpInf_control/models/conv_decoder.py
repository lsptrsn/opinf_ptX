import torch
import torch.nn as nn


class ConvDecoder(nn.Module):
    """
    Convolution-based decoder with flexible normalization, dropout, and activation.
    Designed for POD-CNN reconstruction tasks.
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dims=[64, 512],
        conv_channels=[32, 16, 8, 2],
        kernel_sizes=[3, 3, 3, 5],
        strides=[2, 2, 2, 2],
        paddings=[1, 1, 1, 2],
        dropout_linear=0.0,
        dropout_conv=0.0,
        norm_type: str = "batch",   # "batch", "layer", or "none"
        activation_fn: str = "silu",
    ):
        super().__init__()
        self.norm_type = norm_type.lower()
        self.dropout_linear = dropout_linear
        self.dropout_conv = dropout_conv
        self.activation = self._get_activation(activation_fn)

        # === Linear (fully connected) network ===
        print(self.activation)
        layers = []
        in_features = 2 * latent_dim
        for out_features in hidden_dims:
            block = [nn.Linear(in_features, out_features)]
            if self.norm_type == "batch":
                block.append(nn.BatchNorm1d(out_features))
            elif self.norm_type == "layer":
                block.append(nn.LayerNorm(out_features))
            block.append(self.activation)
            if dropout_linear > 0:
                block.append(nn.Dropout(dropout_linear))
            layers.extend(block)
            in_features = out_features
        self.linear_net = nn.Sequential(*layers)

        # Prepare for reshaping into (channels, spatial)
        self.initial_channels = hidden_dims[0]
        self.initial_spatial = hidden_dims[-1] // self.initial_channels

        # === Convolutional transpose network ===
        conv_blocks = []
        in_channels = self.initial_channels
        for i, out_channels in enumerate(conv_channels):
            conv = nn.ConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                padding=paddings[i],
                output_padding=1,
            )
            block = [conv]
            if self.norm_type == "batch":
                block.append(nn.BatchNorm1d(out_channels))
            elif self.norm_type == "layer":
                block.append(nn.GroupNorm(1, out_channels))  # instance-like norm
            if i < len(conv_channels) - 1:
                block.append(self.activation)
                if dropout_conv > 0:
                    block.append(nn.Dropout(dropout_conv))
            conv_blocks.append(nn.Sequential(*block))
            in_channels = out_channels
        self.conv_net = nn.ModuleList(conv_blocks)

        # === Final projection layer ===
        final_size = self._calculate_output_size()
        self.final_proj = nn.Linear(final_size, output_dim)

        # Optional upsampling before conv layers
        self.upsample = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)

        # Initialize weights
        self.apply(self._init_weights)
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.conv_channels = conv_channels
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.paddings = paddings
        self.dropout_linear = dropout_linear
        self.dropout_conv = dropout_conv
        self.norm_type = norm_type
        self.activation_fn_name = activation_fn

    def _get_activation(self, name: str):
            """Return activation module based on string name."""
            name = name.lower()
            if name == "relu":
                return nn.ReLU()
            elif name == "leakyrelu":
                return nn.LeakyReLU(0.01)
            elif name == "elu":
                return nn.ELU()
            elif name == "selu":
                return nn.SELU()
            elif name == "silu" or name == "swish":
                return nn.SiLU()
            elif name == "gelu":
                return nn.GELU()
            elif name == "softplus":
                return nn.Softplus()
            else:
                raise ValueError(f"Unknown activation: {name}")

    def _init_weights(self, m):
        """Custom weight initialization."""
        if isinstance(m, (nn.Linear, nn.ConvTranspose1d)):
            nn.init.kaiming_uniform_(m.weight, a=0.01, nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _calculate_output_size(self):
        """Compute flattened size after all conv-transpose layers."""
        size = self.initial_spatial
        size = int(size * 2)  # Upsample
        for conv in self.conv_net:
            c = conv[0]  # ConvTranspose1d is always first in the block
            size = (size - 1) * c.stride[0] + c.kernel_size[0] - 2 * c.padding[0] + c.output_padding[0]
        return size * self.conv_net[-1][0].out_channels

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass from latent vector z to reconstructed signal."""
        # Augment latent vector with squared terms
        z2 = z ** 2
        z_all = torch.cat((z, z2), dim=-1)

        # Dense network
        h = self.linear_net(z_all)

        # Reshape into (batch, channels, spatial)
        channels = self.initial_channels
        spatial = h.shape[1] // channels
        h = h.view(h.size(0), channels, spatial)
        h = self.upsample(h)

        # Convolutional transpose network
        for block in self.conv_net:
            h = block(h)

        # Flatten and project to output_dim
        h = h.flatten(1)
        return self.final_proj(h)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decode(z)
