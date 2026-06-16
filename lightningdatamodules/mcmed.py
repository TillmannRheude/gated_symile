import pytorch_lightning as pl
from torch.utils.data import DataLoader

from datasets.mcmed import MCMEDDataset, mcmed_collate_fn
from datasets.mcmed_clinical import MCMEDClinicalDataset, mcmed_clinical_collate_fn

class DataModule_MCMED(pl.LightningDataModule):
    def __init__(
        self, 
        batch_size: int = 128,
        split_nr: int = 1,
        max_waveform_windows: int = 0,
        radiology_model_params: dict = None,
        require_all_modalities: bool = False,
        rad_for_main: str = "future",
        require_both_rad_streams: bool = False,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = 4
        self.split_nr = split_nr
        self.max_waveform_windows = max_waveform_windows
        self.radiology_model_params = radiology_model_params or {}
        self.require_all_modalities = bool(require_all_modalities)
        self.rad_for_main = str(rad_for_main)
        self.require_both_rad_streams = bool(require_both_rad_streams)

    def setup(self, stage):
        """ 
        self.ds_train = MCMEDDataset(
            split_family="chrono",
            split_name="train",
            #data_dir="/sc-resources/dh-mimic/mimic_symile/mcmed_aws/data_preprocessed/future_rads_cv/fold_0/",
            max_waveform_windows=self.max_waveform_windows,
            require_all_modalities=self.require_all_modalities,
            #split_nr=self.split_nr,
        )
        self.ds_val = MCMEDDataset(
            split_family="chrono",
            split_name="val",
            #data_dir="/sc-resources/dh-mimic/mimic_symile/mcmed_aws/data_preprocessed/future_rads_cv/fold_0/",
            max_waveform_windows=self.max_waveform_windows,
            require_all_modalities=self.require_all_modalities,
            #split_nr=self.split_nr,
        )
        self.ds_test = MCMEDDataset(
            split_family="chrono",
            split_name="test",
            #data_dir="/sc-resources/dh-mimic/mimic_symile/mcmed_aws/data_preprocessed/future_rads_cv/fold_0/",
            max_waveform_windows=self.max_waveform_windows,
            require_all_modalities=self.require_all_modalities,
            #split_nr=self.split_nr,
        )
        """

        self.ds_train = MCMEDClinicalDataset(
            split_family="chrono",
            split_name="train",
            #data_dir="/sc-resources/dh-mimic/mimic_symile/mcmed_aws/data_preprocessed/aggregated_memmap_clinical_events/",
            require_all_modalities=self.require_all_modalities,
            rad_for_main=self.rad_for_main,
            require_both_rad_streams=self.require_both_rad_streams,
            #split_nr=self.split_nr,
        )
        self.ds_val = MCMEDClinicalDataset(
            split_family="chrono",
            split_name="val",
            #data_dir="/sc-resources/dh-mimic/mimic_symile/mcmed_aws/data_preprocessed/aggregated_memmap_clinical_events/",
            require_all_modalities=self.require_all_modalities,
            rad_for_main=self.rad_for_main,
            require_both_rad_streams=self.require_both_rad_streams,
            #split_nr=self.split_nr,
        )
        self.ds_test = MCMEDClinicalDataset(
            split_family="chrono",
            split_name="test",
            #data_dir="/sc-resources/dh-mimic/mimic_symile/mcmed_aws/data_preprocessed/aggregated_memmap_clinical_events/",
            require_all_modalities=self.require_all_modalities,
            rad_for_main=self.rad_for_main,
            require_both_rad_streams=self.require_both_rad_streams,
            #split_nr=self.split_nr,
        )

    def train_dataloader(self):
        return DataLoader(
            self.ds_train, batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=mcmed_clinical_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_val, batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=mcmed_clinical_collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.ds_test, batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=mcmed_clinical_collate_fn,
        )
