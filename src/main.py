import os
import torch
import pretrainedmodels
import numpy as np
import pandas as pd
import albumentations

#from apex import amp

import torch.nn as nn
from sklearn import metrics
from torch.nn import functional as F

from classification import ClassificationDataLoader
from early_stopping import EarlyStopping
from engine import Engine

class SEResNext50_32x4d(nn.Module):
    def __init__(self, pretrained="imagenet"):
        super(SEResNext50_32x4d, self).__init__() #Super class is to access base class without specifing class name
        self.model = pretrainedmodels.__dict__["se_resnext50_32x4d"](pretrained=pretrained)
        self.out = nn.Linear(2048, 1) #last Linear layer with  inputs in_features = 2048 by default

    def forward(self, image, target):
        bs, _, _, _ = image.shape # _ are for channel, height and width, We just need batch size
        x = self.model.features(image) # here image is batch of images
        x = F.adaptive_avg_pool2d(x, 1) # need adaptive avg pooling to support batch size
        x = x.reshape(bs, -1) #reshape batchsize
        out = self.out(x)
        loss = nn.BCEWithLogitsLoss()(
            out, target.reshape(-1, 1).type_as(out)
            )
        return out, loss


def train(fold):
    training_data_path = "/content/drive/MyDrive/Data_science_projects/Melanoma/dataset/jpeg_output/train/"
    model_path = "/content/drive/MyDrive/Data_science_projects/Melanoma/melanoma-deep-learning"
    df = pd.read_csv("/content/drive/MyDrive/Data_science_projects/Melanoma/dataset/train_folds.csv")

    device = "cuda"
    epochs = 50
    train_bs = 32
    valid_bs = 16
    mean = (0.485, 0.456, 0.406) # came for model mean
    std = (0.229, 0.224, 0.225)

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # To normalize the images
    train_aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
        ]
    )

    valid_aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
        ]
    )

    train_images = df_train.image_name.values.tolist()
    train_images = [os.path.join("training_data_path", i + '*.jpg' ) for i in train_images] # list of image paths
    train_targets = df_train.target.values

    valid_images = df_valid.image_name.values.tolist()
    valid_images = [os.path.join("training_data_path", i + '*.jpg' ) for i in valid_images] # list of image paths
    valid_targets = df_valid.target.values

    train_dataset = ClassificationDataLoader(
        image_paths = train_images,
        targets= train_targets,
        resize= None,
        augmentations = train_aug
    ).fetch(batch_size=train_bs, num_workers=2, drop_last=False, shuffle=True, tpu=False)


    valid_dataset = ClassificationDataLoader(
        image_paths = valid_images,
        targets= valid_targets,
        resize= None,
        augmentations = valid_aug
    ).fetch(batch_size=valid_bs, num_workers=2, drop_last=False, shuffle=False, tpu=False)
    # here shuffle is False as we need to use targets used from classification file

    model = SEResNext50_32x4d(pretrained="imagenet")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        patience=3, 
        mode="max"
        ) # using mode as max as we will using auc

    """
    model, optimizer = amp.initialize(
        model,
        optimizer,
        opt_level='01',
        verbosity=0) 
        # this is not required and used increase speed, can be skipped """

    es = EarlyStopping(patience=5, mode="max")
    for epoch in range(epochs):
        training_en = Engine(model, optimizer, device, fp16=True)
        training_loss = training_en.train(train_loader)

        predictions, valid_loss = training_en.evaluate(
            train_loader, 
            return_predictions=True)

        predictions = np.vstack((predictions)).ravel() #flatten the 1D array
        auc = metrics.roc_auc_score(valid_targets, predictions)
        scheduler.step(auc)
        print(f"epoch={epoch}, auc={auc}")
        es(auc, model, os.path.join(model_path, f"model{fold}.bin"))
        if es.early_stop:
            print("early stopping")
            break

def predict(fold):
    test_data_path = "/content/drive/MyDrive/Data_science_projects/Melanoma/dataset/jpeg_output/test/"
    model_path = "/content/drive/MyDrive/Data_science_projects/Melanoma/melanoma-deep-learning"
    df_test = pd.read_csv("/content/drive/MyDrive/Data_science_projects/Melanoma/dataset/test.csv")
    df_test.loc[:, target] = 0
    

    device = "cuda"
    epochs = 50
    test_bs = 16
    mean = (0.485, 0.456, 0.406) # came for model mean
    std = (0.229, 0.224, 0.225)

   

    # To normalize the images

    test_aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
        ]
    )

    test_images = df_test.image_name.values.tolist()
    test_images = [os.path.join("test_data_path", i + '*.jpg' ) for i in test_images] # list of image paths
    test_targets = df_test.target.values #Test targets are of no user but we need them for classification loader

    test_dataset = ClassificationDataLoader(
        image_paths = test_images,
        targets= test_targets,
        resize= None,
        augmentations = test_aug
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=test_bs,
        shuffle=False,
        num_workers=2
    )

     # here shuffle is False as we need to use targets used from classification file

     # We need to add model weights

    model = SEResNext50_32x4d(pretrained="imagenet")
    model.load_state_dict(torch.load(os.path.join(model_path, f"model{fold}.bin")))
    model.to(device)

    predictions = Engine.predict(test_loader)

    return np.vstack((predictions)).ravel()

if __name__ == '__main__':
    train(fold=0)









