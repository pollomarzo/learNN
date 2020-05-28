import pandas as pd
import numpy as np
import torch
from torchtext.data.dataset import Dataset
from torchtext.data.dataset import check_split_ratio
from torchtext.data import Example


class FakeNewsDataset(Dataset):
    """
    Implements a dataset that reads from a clean data csv file. 

    Extremely simple for now, may move the cleaning section here in the future
    """

    def __init__(self, csv_path, text_field, label_field):
        # full_data is a DataFrame with data and labels fields
        full_data = pd.read_csv(csv_path)
        data = np.asarray(full_data.data)
        labels = np.asarray(full_data.labels)

        fields = [('text', text_field), ('label', label_field)]
        examples = []
        for i, e in enumerate(data):
            examples.append(Example.fromlist([e, labels[i]], fields))

        super(FakeNewsDataset, self).__init__(examples, fields)
