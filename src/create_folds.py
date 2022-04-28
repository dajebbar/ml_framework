import numpy as np
import pandas as pd
from sklearn import model_selection

if __name__ == '__main__':
    df = pd.read_csv('input/train.csv')
    df['kfold'] = -1

    df = df.sample(frac=1).reset_index(drop=True)

    kf = model_selection.StratifiedKFold(n_splits=5)
    y = df.target.values

    for f_, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        print(len(t_), len(v_))
        df.loc[v_, 'kfold'] = f_

    df.to_csv('input/train_folds.csv', index=False)
