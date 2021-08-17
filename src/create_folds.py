import pandas as pd
import os
from sklearn import model_selection

if __name__ == "__main__":
    input_path = "/content/drive/MyDrive/Data_science_projects/Melanoma/dataset/"
    df = pd.read_csv(os.path.join(input_path,"train.csv"))
    df["kfold"] = -1 #adding kfold column and assiging -1 value
    df = df.sample(frac=1).reset_index(drop=True) #shuffling the dataframe
    y = df.target.values
    kf = model_selection.StratifiedKFold(n_splits=5)

    for fold_, (_, _) in enumerate(kf.split(X=df, y=y)):
        df.loc[:, "kfold"] = fold_
    df.to_csv(os.path.join(input_path, "train_folds.csv"), index=False)
