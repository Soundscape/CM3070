import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split


class ScheduleClassifierConfig(BaseModel):
    """Defines the schedule classifier's parameters"""
    lr: float = Field(...)
    input_dim: int = Field(...)
    output_dim: int = Field(...)
    fc1_dim: int = Field(...)
    fc2_dim: int = Field(...)
    fc3_dim: int = Field(...)
    fc4_dim: int = Field(...)
    fc5_dim: int = Field(...)
    fc6_dim: int = Field(...)
    dropout: int = Field(...)


class ScheduleClassifier(pl.LightningModule):
    """The schedule classifier model"""

    def __init__(self, config: dict) -> None:
        super().__init__()

        self.config = ScheduleClassifierConfig.parse_obj(config)

        self.input = nn.Linear(self.config.input_dim, self.config.fc1_dim)
        self.fc1 = nn.Linear(self.config.fc1_dim, self.config.fc2_dim)
        self.fc2 = nn.Linear(self.config.fc2_dim, self.config.fc3_dim)
        self.fc3 = nn.Linear(self.config.fc3_dim, self.config.fc4_dim)
        self.fc4 = nn.Linear(self.config.fc4_dim, self.config.fc5_dim)
        self.fc5 = nn.Linear(self.config.fc5_dim, self.config.fc6_dim)
        self.fc6 = nn.Linear(self.config.fc6_dim, self.config.output_dim)

    def forward(self, x):
        batch_size, _ = x.size()

        x = x.view(batch_size, -1)
        x = F.relu(self.input(x))
        x = F.relu(self.fc1(x))
        x = F.dropout(x, self.config.dropout)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.softmax(x, dim=1)

        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        return optimizer

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def accuracy(self, logits, labels):
        _, predicted = torch.max(logits.data, 1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / len(labels)

        return torch.tensor(accuracy)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        accuracy = self.accuracy(logits, y)

        self.log('train_loss', loss)
        self.log('train_accuracy', accuracy)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        accuracy = self.accuracy(logits, y)

        return {'val_loss': loss, 'val_accuracy': accuracy}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_accuracy = torch.stack([x['val_accuracy'] for x in outputs]).mean()

        self.log('val_loss', avg_loss)
        self.log('val_accuracy', avg_accuracy)

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        accuracy = self.accuracy(logits, y)

        return {'test_loss': loss, 'test_accuracy': accuracy}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_accuracy = torch.stack([x['test_accuracy']
                                   for x in outputs]).mean()

        self.log('test_loss', avg_loss)
        self.log('test_accuracy', avg_accuracy)


class ScheduleDataModuleConfig(BaseModel):
    """Defines the schedule classifier's data module parameters"""
    file_path: str = Field(...)
    batch_size: int = Field(...)
    train_frac: float = Field(...)
    test_frac: float = Field(...)
    val_frac: float = Field(...)
    num_workers: int = Field(...)


class ScheduleDataModule(pl.LightningDataModule):
    """The data module for use with a schedule classifier"""

    def __init__(self, config: dict) -> None:
        super().__init__()

        self.config = ScheduleDataModuleConfig.parse_obj(config)
        self.ds = None
        self.train, self.test, self.val = None, None, None

    def setup(self, stage: str) -> None:
        df = pd.read_csv(self.config.file_path)
        df = df.sample(frac=1.0)

        y = torch.tensor(df.iloc[:, 1].values, dtype=torch.long)
        x = torch.tensor(df.iloc[:, 2:].values, dtype=torch.float32)

        self.ds = TensorDataset(x, y)
        splits = [
            self.config.train_frac,
            self.config.test_frac,
            self.config.val_frac
        ]
        self.train, self.test, self.val = random_split(self.ds, splits)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.config.batch_size, num_workers=self.config.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.config.batch_size, num_workers=self.config.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.config.batch_size, num_workers=self.config.num_workers)
