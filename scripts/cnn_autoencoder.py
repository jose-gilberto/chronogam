from typing import Any
import numpy as np
import pandas as pd
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class LeakySineLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Performs the LeakySineLU activation.
        
        $$
        \sigma(x) = \begin{cases}
            \sin(x)^{2} + x & \text{if} x \ge 0 \\
            0.1 (\sin(x)^{2} + x) & \text{otherwise.}
        \end{cases}
        $$

        Args:
            x (torch.Tensor): The input tensor that will be performed the calculus.

        Returns:
            torch.Tensor: The output tensor with the activation applied.
        """
        # return torch.max(0.1 * (torch.sin(x)**2 + x), torch.sin(x)**2 + x)
        return torch.sin(x) ** 2 + x


class GlobalAveragePool(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mean(x, dim=1)


class Upscale(nn.Module):
    def __init__(self, out_channels: int, sequence_length: int) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.sequence_length = sequence_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape((x.size(0), self.out_channels, self.sequence_length))


def init_weights(m: nn.Module):
    if isinstance(m, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
        torch.nn.init.xavier_uniform_(m.weight)

class BaseDataset(Dataset):
    def __init__(self, x, y) -> None:
        super().__init__()
        self.x = x
        self.y = y
        
        assert len(self.x) == len(self.y)
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index: int):
        return torch.from_numpy(self.x[index]).float(), torch.from_numpy(np.array([self.y[index]]))
    
    
class CNNAutoencoder(pl.LightningModule):
    def __init__(self, in_features: int, latent_dim: int = 2) -> None:
        super().__init__()
        
        self.in_features = in_features
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, padding='same', bias=False),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding='same', bias=False),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, padding='same', bias=False),
            nn.LeakyReLU(),
            GlobalAveragePool(),
            nn.Linear(in_features=in_features, out_features=latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=in_features * 16, bias=False),
            Upscale(sequence_length=in_features, out_channels=16),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=16, out_channels=32, kernel_size=3, padding=3//2, bias=False),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=5, padding=5//2, bias=False),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=16, out_channels=1, kernel_size=7, padding=7//2, bias=False),
        )
        
        self.criteria = nn.MSELoss()
        
    def forward(self, x: torch.Tensor) -> tuple:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z
    
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x, y = batch
        x_hat, z = self(x)
        
        loss = self.criteria(x_hat, x)
        self.log('training loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        
        return loss
    
    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(self.parameters(), lr=1e-3)


UCR_DATASETS = [
    # 'Adiac',
    # 'ArrowHead',
    # 'Beef',
    # 'BeetleFly',
    # 'BirdChicken',
    # 'Car',
    # 'CBF',
    # 'ChlorineConcentration',
    # 'CinCECGTorso',
    # 'Coffee',
    # 'Computers',
    # 'CricketX',
    # 'CricketY',
    # 'CricketZ',
    # 'DiatomSizeReduction',
    # 'DistalPhalanxOutlineAgeGroup',
    # 'DistalPhalanxOutlineCorrect',
    # 'DistalPhalanxTW',
    # 'Earthquakes',
    # 'ECG200',
    'ECG5000',
    # 'ECGFiveDays',
    # 'ElectricDevices',
    # 'FaceAll',
    # 'FaceFour',
    # 'FacesUCR',
    # 'FiftyWords',
    # 'Fish',
    # 'FordA',
    # 'FordB',
    # 'GunPoint',
    # 'Ham',
    # 'HandOutlines',
    # 'Haptics',
    # 'Herring',
    # 'InlineSkate',
    # 'InsectWingbeatSound',
    # 'ItalyPowerDemand',
    # 'LargeKitchenAppliances',
    # 'Lightning2',
    # 'Lightning7',
    # 'Mallat',
    # 'Meat',
    # 'MedicalImages',
    # 'MiddlePhalanxOutlineAgeGroup',
    # 'MiddlePhalanxOutlineCorrect',
    # 'MiddlePhalanxTW',
    # 'MoteStrain',
    # 'NonInvasiveFetalECGThorax1',
    # 'NonInvasiveFetalECGThorax2',
    # 'OliveOil',
    # 'OSULeaf',
    # 'PhalangesOutlinesCorrect',
    # 'Phoneme',
    # 'Plane',
    # 'ProximalPhalanxOutlineAgeGroup',
    # 'ProximalPhalanxOutlineCorrect',
    # 'ProximalPhalanxTW',
    # 'RefrigerationDevices',
    # 'ScreenType',
    # 'ShapeletSim',
    # 'ShapesAll',
    # 'SmallKitchenAppliances',
    # 'SonyAIBORobotSurface1',
    # 'SonyAIBORobotSurface2',
    # 'StarLightCurves',
    # 'Strawberry',
    # 'SwedishLeaf',
    # 'Symbols',
    # 'SyntheticControl',
    # 'ToeSegmentation1',
    # 'ToeSegmentation2',
    # 'Trace',
    # 'TwoLeadECG',
    # 'TwoPatterns',
    # 'UWaveGestureLibraryAll',
    # 'UWaveGestureLibraryX',
    # 'UWaveGestureLibraryY',
    # 'UWaveGestureLibraryZ',
    # 'Wafer',
    # 'Wine',
    # 'WordSynonyms',
    # 'Worms',
    # 'WormsTwoClass',
    # 'Yoga',
    # 'ACSF1',
    # 'BME',
    # 'Chinatown',
    # 'Crop',
    # 'EOGHorizontalSignal',
    # 'EOGVerticalSignal',
    # 'EthanolLevel',
    # 'FreezerRegularTrain',
    # 'FreezerSmallTrain',
    # 'Fungi',
    # 'GunPointAgeSpan',
    # 'GunPointMaleVersusFemale',
    # 'GunPointOldVersusYoung',
    # 'HouseTwenty',
    # 'InsectEPGRegularTrain',
    # 'InsectEPGSmallTrain',
    # 'MixedShapesRegularTrain',
    # 'MixedShapesSmallTrain',
    # 'PigAirwayPressure',
    # 'PigArtPressure',
    # 'PigCVP',
    # 'PowerCons',
    # 'Rock',
    # 'SemgHandGenderCh2',
    # 'SemgHandMovementCh2',
    # 'SemgHandSubjectCh2',
    # 'SmoothSubspace',
    # 'UMD'
]

for dataset in UCR_DATASETS:
    print(f'Starting experiments with {dataset} dataset...')
    # Load the data from .tsv files
    train_data = np.genfromtxt(f'../data/ucr/{dataset}/{dataset}_TRAIN.tsv')
    x_train, y_train = train_data[:, 1:], train_data[:, 0]
    
    test_data = np.genfromtxt(f'../data/ucr/{dataset}/{dataset}_TEST.tsv')
    x_test, y_test = test_data[:, 1:], test_data[:, 0]

    # Filter samples from positive label
    x_train_ = x_train
    y_train_ = y_train

    # Apply z normalization
    std_ = x_train_.std(axis=1, keepdims=True)
    std_[std_ == 0] = 1.0
    x_train_ = (x_train_ - x_train_.mean(axis=1, keepdims=True)) / std_
    
    std_ = x_test.std(axis=1, keepdims=True)
    std_[std_ == 0] = 1.0
    x_test_ = (x_test - x_test.mean(axis=1, keepdims=True)) / std_
    
    x_train_ = np.expand_dims(x_train_, axis=1)
    x_test_ = np.expand_dims(x_test, axis=1)

    train_set = BaseDataset(x=x_train_, y=y_train_)
    test_set = BaseDataset(x=x_test_, y=y_test)
    
    train_loader = DataLoader(train_set, batch_size=16)
    test_loader = DataLoader(test_set, batch_size=16)
    
    autoencoder = CNNAutoencoder(in_features=x_train_.shape[-1], latent_dim=2)
    trainer = pl.Trainer(
            max_epochs=1000,
            accelerator='gpu',
            devices=-1,
    )
    
    trainer.fit(autoencoder, train_loader)
    
    latent_representations = []
    
    for x, y in train_loader:
        x_hat, z = autoencoder(x)
        latent_representations.extend(z.detach().tolist())

    latent_representations = np.array(latent_representations)

    plt.figure(figsize=(6, 6))
    
    plt.scatter(latent_representations[:,0], latent_representations[:,1], color='royalblue', marker='o')
    
    plt.xlabel('z_1')
    plt.ylabel('z_2')

    plt.tight_layout()

    plt.savefig('latent_space.pdf')

