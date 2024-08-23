import os

from src.Probing.probe import linear_probe
from src.Probing.probe_trainer import probe_trainer
from src.Probing.probe_trainer import probing_dataset


class POS_probe_trainer(probe_trainer):
    def __init__(self, model: linear_probe, device, num_shift_pos=0):
        super().__init__(model, device, "POS")
        self.num_shift_pos = num_shift_pos

    def load_dataset(self, dataset_purpose, subset_num):
        dataset = POS_probing_dataset(
            self.device,
            self.dataset_name,
            dataset_purpose=dataset_purpose,
            subset_num=subset_num,
            num_shift_pos=self.num_shift_pos,
        )

        return dataset

    def save_model(self):
        save_dir = os.path.join(
            os.getcwd(),
            "data",
            "probe_datasets",
            self.dataset_name,
            f"shift{self.num_shift_pos}",
        )
        super().save_model(save_dir)

        return

    def load_model(self):
        load_dir = os.path.join(
            os.getcwd(),
            "data",
            "probe_datasets",
            self.dataset_name,
            f"shift{self.num_shift_pos}",
        )

        super().load_model(load_dir)

        return


class POS_probing_dataset(probing_dataset):
    def __init__(
        self,
        device,
        dataset_name: str,
        dataset_purpose: str,
        subset_num=0,
        dataset_dir=None,
        num_shift_pos=0,
    ):
        self.num_shift_pos = num_shift_pos
        if dataset_dir is None:
            dataset_dir = os.path.join(
                os.getcwd(),
                "data",
                "probe_datasets",
                dataset_name,
                f"shift{self.num_shift_pos}",
                dataset_purpose,
            )
        super().__init__(device, dataset_purpose, subset_num, dataset_dir)
