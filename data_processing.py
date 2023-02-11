"""Contains ETL tools for ImageNet"""
from typing import List

import pandas as pd
import tensorflow as tf


class DatasetLoader:
    def __init__(self, num_of_classes: int, output_dir: str):
        self.num_of_classes = num_of_classes
        self.output_dir = output_dir

        self.dataset = self.download_imagenet()
        self.dataset = self.select_classes()

    def download_imagenet(self) -> pd.DataFrame:
        pass

    def select_classes(self) -> pd.DataFrame:
        pass

    def store_dataset_by_classes(self) -> None:
        pass

    def load_tf_datasets(self) -> List[tf.data.Dataset]:
        pass
