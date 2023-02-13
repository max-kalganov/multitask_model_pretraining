"""Contains ETL tools for ImageNet"""
import os
from typing import List, Tuple

import numpy as np
import tensorflow as tf


class DatasetLoader:
    def __init__(self, classes_to_leave: List[int], output_dir: str, download_dataset: bool = True):
        self.classes_to_leave = classes_to_leave
        self.output_dir = f"{output_dir}_{'_'.join(map(str, classes_to_leave))}"
        self.train_data = None, None
        self.test_data = None, None

        if download_dataset:
            self.train_data, self.test_data = self.download_dataset()
            self.train_data, self.test_data = self.select_classes()

    def get_classes(self) -> List[int]:
        train_labels = self.train_data[1]
        return sorted(set(train_labels))

    def get_full_dataset_path(self) -> str:
        return os.path.join(self.output_dir, 'full_dataset')

    def get_dataset_by_classes_path(self) -> str:
        return os.path.join(self.output_dir, 'by_classes')

    @staticmethod
    def download_dataset() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        mnist = tf.keras.datasets.mnist
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        return (train_images, train_labels), (test_images, test_labels)

    def select_classes(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        def __select_samples(x, y) -> Tuple[np.ndarray, np.ndarray]:
            condition = sum([y == selected_class for selected_class in self.classes_to_leave]).astype('bool')
            return x[condition], y[condition]

        train_images, train_labels = self.train_data
        test_images, test_labels = self.test_data

        return __select_samples(train_images, train_labels), __select_samples(test_images, test_labels)

    def _get_full_dataset_train_path(self):
        return os.path.join(self.get_full_dataset_path(), 'train_x.npy'),\
               os.path.join(self.get_full_dataset_path(), 'train_y.npy')

    def _get_full_dataset_test_path(self):
        return os.path.join(self.get_full_dataset_path(), 'test_x.npy'),\
               os.path.join(self.get_full_dataset_path(), 'test_y.npy')

    def store_full_dataset(self) -> None:
        def __store(x, y, x_path, y_path):
            np.save(x_path, x)
            np.save(y_path, y)
        os.makedirs(self.get_full_dataset_path(), exist_ok=True)
        __store(*self.train_data, *self._get_full_dataset_train_path())
        __store(*self.test_data, *self._get_full_dataset_test_path())

    def load_full_dataset(self):
        def __load(x_path, y_path):
            x = np.load(x_path)
            y = np.load(y_path)
            return x, y

        assert os.path.exists(self.output_dir)

        self.train_data = __load(*self._get_full_dataset_train_path())
        self.test_data = __load(*self._get_full_dataset_test_path())

    def store_dataset_by_classes(self) -> None:
        pass

    def load_tf_datasets(self) -> List[tf.data.Dataset]:
        pass


def test_full_dataset_loader():
    dl = DatasetLoader([1, 2, 3], 'data')
    dl.store_full_dataset()
    del dl

    dl2 = DatasetLoader([1, 2], 'data', download_dataset=False)
    assert dl2.train_data == (None, None), "Not empty data"

    try:
        dl2.load_full_dataset()
        raise ValueError("Unexpected behaviour")
    except AssertionError:
        pass
    del dl2

    dl3 = DatasetLoader([1, 2, 3], 'data', download_dataset=False)
    dl3.load_full_dataset()
    assert dl3.get_classes() == [1, 2, 3], "incorrect classes loaded"


if __name__ == '__main__':
    dl3 = DatasetLoader([1, 2, 3], 'data', download_dataset=False)
    dl3.load_full_dataset()
    print(dl3.train_data[0].shape)
