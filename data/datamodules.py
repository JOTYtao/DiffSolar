import logging
from data.datasets import SISDataset
_LOG = logging.getLogger(__name__)
_LOG.setLevel(logging.DEBUG)
from typing import Any, Dict, Optional
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torch
from pathlib import Path
import pandas as pd
class SIS_DataModule(LightningDataModule):
    def __init__(
        self,
        dataset: Dict[str, Any],
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.dataset_config = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def _validate_config(self):
        required_keys = ['data_path', 'years', 'input_len', 'pred_len']
        for key in required_keys:
            if key not in self.dataset_config:
                raise ValueError(f"Missing required config key: {key}")
        data_path = Path(self.dataset_config['data_path'])
        if not data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {data_path}")
        _LOG.info(f"Data configuration validated successfully")
        _LOG.debug(f"Dataset config: {self.dataset_config}")

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`."""
        _LOG.info(f"Setting up data module for stage: {stage}")
        try:
            if not self.data_train and not self.data_val and not self.data_test:
                _LOG.debug("Initializing datasets...")
                self.data_train = SISDataset(
                    **self.dataset_config,
                    mode="train",
                )
                _LOG.info(f"Training dataset initialized with {len(self.data_train)} samples")
                self.data_val = SISDataset(
                    **self.dataset_config,
                    mode="val",
                )
                _LOG.info(f"Validation dataset initialized with {len(self.data_val)} samples")
                self.data_test = SISDataset(
                    **self.dataset_config,
                    mode="test",
                )
                _LOG.info(f"Test dataset initialized with {len(self.data_test)} samples")
        except Exception as e:
            _LOG.error(f"Error setting up datasets: {str(e)}")
            raise

    def train_dataloader(self):
        if self.data_train is None:
            _LOG.info("Train dataset not initialized, calling setup()")
            self.setup(stage='fit')

        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True
        )

    def val_dataloader(self):
        if self.data_val is None:
            _LOG.info("Validation dataset not initialized, calling setup()")
            self.setup(stage='fit')

        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        if self.data_test is None:
            _LOG.info("Test dataset not initialized, calling setup()")
            self.setup(stage='test')

        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    @property
    def num_samples(self):
        """Returns the number of samples in all datasets."""
        return {
            'train': self.num_train_samples,
            'val': self.num_val_samples,
            'test': self.num_test_samples
        }

    @property
    def num_train_samples(self):
        if self.data_train is None:
            self.setup(stage='fit')
        return len(self.data_train) if self.data_train else 0

    @property
    def num_val_samples(self):
        if self.data_val is None:
            self.setup(stage='fit')
        return len(self.data_val) if self.data_val else 0

    @property
    def num_test_samples(self):
        if self.data_test is None:
            self.setup(stage='test')
        return len(self.data_test) if self.data_test else 0


def verify_dataloader_shapes(config):
    # 创建 datamodule
    datamodule = SIS_DataModule(
        dataset=config,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory']
    )
    datamodule.setup()
    print(f"Train samples: {datamodule.num_train_samples}")
    print(f"Validation samples: {datamodule.num_val_samples}")
    print(f"Test samples: {datamodule.num_test_samples}")
    test_loader = datamodule.test_dataloader()
    save_dir = os.path.join(config['data_path'], "time_batches")
    os.makedirs(save_dir, exist_ok=True)
    all_targets = []
    for batch in test_loader:

        target_data = batch['target'].numpy() if isinstance(batch['target'], torch.Tensor) else batch['target']
        all_targets.append(target_data)

    # 拼接所有时间数据
    all_times = np.concatenate(all_targets, axis=0)  # 拼接成 (N, pred_len, H, W) 或 (N, pred_len)
    time_save_path = os.path.join(save_dir, "test_target.npy")
    np.save(time_save_path, all_times)
    # time_data = np.load("E:/research/my_code/solar_flow/data/time_batches/test_time.npy")
    # time_data = time_data[:, 0]
    # time_data_standard = pd.to_datetime(time_data, unit='s')
    # save_path = "E:/research/my_code/solar_flow/data/time_batches/train_time_standard.csv"
    # time_data_standard.to_frame(name="time").to_csv(save_path, index=False)



if __name__ == '__main__':
    dataset_config = {
        "data_path": "E:/research/my_code/solar_flow/data",
        "years": {
            "train": [2017, 2018, 2019, 2020],
            "val": [2021],
            "test": [2022]
        },
        "input_len": 8,
        "pred_len": 8,
        "stride": 1,
        "forecast": True,
        "use_possible_starts": True,
        "batch_size": 16,
        "num_workers": 10,
        "pin_memory": True
    }

    verify_dataloader_shapes(dataset_config)