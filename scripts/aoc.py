from typing import Any, Optional, Tuple, Union
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl


class AOCAutoEncoder(pl.LightningModule):
    def __init__(self, in_channels: int, project_channels: int, hidden_size: int, window_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.project_channels = project_channels
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.encoder = nn.LSTM(
            self.in_channels,
            self.hidden_size,
            batch_first=True,
            num_layers=self.num_layers,
            bias=False,
            dropout=self.dropout
        )
        
        self.decoder = nn.LSTM(
            self.in_channels,
            self.hidden_size,
            batch_first=True,
            num_layers=self.num_layers,
            bias=False,
            dropout=self.dropout
        )
        
        self.output_layer = nn.Linear(self.hidden_size, self.in_channels)
        self.projection_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2, bias=False),
            nn.BatchNorm1d(self.hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size // 2, self.project_channels, bias=False),
        )
        
        self.center = torch.zeros(self.project_channels, device=self.device)

    def init_hidden_state(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros((self.num_layers, batch_size, self.hidden_size)).to(self.device)
        c = torch.zeros((self.num_layers, batch_size, self.hidden_size)).to(self.device)
        return h, c

    def forward(self, x: torch.Tensor, return_latent: bool = True, training: bool = True) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        encoder_hidden = self.init_hidden_state(x.shape[0])
        _, encoder_hidden = self.encoder(x.float(), encoder_hidden)
        decoder_hidden = encoder_hidden

        output = torch.zeros(x.shape).to(self.device)

        for i in reversed(range(x.shape[1])):
            output[:, i, :] = self.output_layer(decoder_hidden[0][0, :])
            if training:
                _, decoder_hidden = self.decoder(x[:, i].unsqueeze(1).float(), decoder_hidden)
            else:
                _, decoder_hidden = self.decoder(output[:, i].unsqueeze(1), decoder_hidden)
                
        hidden = self.projection_head(encoder_hidden[1][-1])

        return (output, hidden) if return_latent else output

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4, betas=(0.9, 0.99), weight_decay=5e-4)
        early_stopping = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        return [optimizer], [early_stopping]

    def init_center(self, dataloader: DataLoader, eps: float = 0.1) -> torch.Tensor:
        n_samples = 0
        c = self.center

        self.eval()
        with torch.no_grad():
            for (x, y) in dataloader:
                x = x.float().to(self.device)
                output, hidden = self(x)
                n_samples += x.shape[0]
                all_features = hidden
                c += torch.sum(all_features, dim=0)

        c /= n_samples

        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        self.center = c

    def training_step(self) -> STEP_OUTPUT:
        return super().training_step()
    
    def test_step(self) -> STEP_OUTPUT:
        return super().test_step()