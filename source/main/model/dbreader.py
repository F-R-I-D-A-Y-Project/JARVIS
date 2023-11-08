import pathlib
from typing import Self
from torch.utils.data import Dataset, DataLoader


class DBManager:
    def __init__(self, path: pathlib.Path, *,
                 dataset: str='dataset.csv', 
                 command_dataset: str='commands.csv') -> None:
        self.__dataset_path = path
        self.__dataset = dataset
        self.__cmd_dataset = command_dataset

    @property
    def dataset_path(self) -> pathlib.Path:
        '''
            This property returns the path to the dataset.
        '''
        return self.__dataset_path
    
    @property
    def dataset(self) -> Dataset:
        '''
            This property returns the dataset.
        '''
        return self.__dataset
    
    @property
    def cmd_dataset(self) -> Dataset:
        '''
            This property returns the command dataset.
        '''
        return self.__cmd_dataset
    
    def split(self: Self, train_size: float=0.8) -> None:
        '''
            This method splits the dataset into train, test and validation sets.
        '''