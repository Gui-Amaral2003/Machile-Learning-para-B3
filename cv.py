from sklearn.model_selection import BaseCrossValidator
import numpy as np
import pandas as pd
class PurgedTimeSeriesSplit(BaseCrossValidator):
    def __init__(self, n_splits = 5, gap_days = 5):
        self.n_splits = n_splits
        self.gap_days = gap_days

    def split(self, X, y = None, groups = None):
        dates = pd.to_datetime(groups)
        unique = np.sort(dates.unique())
        fold_size = len(unique) // (self.n_splits + 1)

        for i in range(1, self.n_splits + 1):
            train_end = unique[i * fold_size]
            val_start = train_end + pd.Timedelta(days=self.gap_days)
            val_end = unique[min((i+1) * fold_size, len(unique) - 1)]

            train_idx = np.where(dates <= train_end)[0]
            val_idx = np.where((dates >= val_start) & (dates <= val_end))[0]

            if len(train_idx) > 0 and len(val_idx) > 0:
                yield train_idx, val_idx
    
    def get_n_splits(self, X = None, y = None, groups = None):
        return self.n_splits