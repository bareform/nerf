import torch
import torch.nn as nn
import torch.nn.functional as F

class NeRF(nn.Module):
    def __init__(
        self,
        depth: int=8,
        W: int=256,
        in_channels: int=3,
        in_channel_views: int=3,
        out_channels: int=4,
        skip_connections: list[int]=[4],
        use_viewdirs: bool=False,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.W = W
        self.in_channels = in_channels
        self.in_channel_views = in_channel_views
        self.out_channels = out_channels
        self.skip_connections = skip_connections
        self.use_viewdirs = use_viewdirs

        self.linear_layers = nn.ModuleList(
            [nn.Linear(self.in_channels, self.W)] +
            [
                nn.Linear(self.W, self.W)
                if i not in self.skip_connections
                else nn.Linear(self.W + self.in_channels, self.W)
                for i in range(self.depth - 1)
            ]
        )

        self.views_linear_layers = nn.ModuleList([
            nn.Linear(self.in_channel_views + self.W, self.W // 2)
        ])

        if self.use_viewdirs:
            self.feature_linear = nn.Linear(self.W, self.W)
            self.alpha_linear = nn.Linear(self.W, 1)
            self.rgb_linear = nn.Linear(self.W // 2, 3)
        else:
            self.output_linear = nn.Linear(self.W, self.out_channels)

    def forward(self, input_pts: torch.Tensor, input_views: torch.Tensor = None) -> torch.Tensor:
        h = input_pts
        for idx, linear_layer in enumerate(self.linear_layers):
            h = linear_layer(h)
            h = F.relu(h)
            if idx in self.skip_connections:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs and input_views is not None:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
            for linear_layer in self.views_linear_layers:
                h = linear_layer(h)
                h = F.relu(h)
            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs