import os
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import forge.src.transform as forge
from sklearn.model_selection import GroupShuffleSplit
import SimpleITK as sitk
import torch


class IMGDataset(Dataset):
    def __init__(self, metadata_file_path, transforms,
                 has_mask=True):
        super(Dataset, self).__init__()
        self.transforms = transforms
        self.has_mask = has_mask
        data = pd.read_csv(metadata_file_path)
        self.data = [dict(x) for _, x in data.iterrows()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        datapoint = self.data[item]
        image_path = datapoint.get('Image')
        mask_path = datapoint.get('Mask', None)
        if self.transforms is not None:
            image, mask = self.transforms(image_path, mask_path)
        return image, mask

    @staticmethod
    def collate(batch):
        images, contours = list(zip(*batch))
        images = torch.stack(images)
        if contours[0] is not None:
            contours = torch.stack(contours)
        else:
            contours = None
        return images, contours


def train_valid_test_split(file_address, out_dir, train_ratio=0.6,
                           test_ratio=0.2, random_state=1):
    assert train_ratio + test_ratio < 1
    data = pd.read_csv(file_address)
    shuffle = GroupShuffleSplit(n_splits=1, test_size=test_ratio,
                                random_state=random_state)
    train_valid_idx, test_idx = next(shuffle.split(data,
                                                   groups=data['PatientID']))
    train_valid_data = data.iloc[train_valid_idx, :]
    test_data = data.iloc[test_idx, :]
    shuffle = GroupShuffleSplit(n_splits=1, train_size=train_ratio,
                                random_state=random_state)
    train_idx, valid_idx = next(shuffle.split(train_valid_data,
                                              groups=train_valid_data['PatientID']))
    train_data = train_valid_data.iloc[train_idx, :]
    valid_data = train_valid_data.iloc[valid_idx, :]
    train_data.to_csv(os.path.join(out_dir, 'train.csv'))
    valid_data.to_csv(os.path.join(out_dir, 'valid.csv'))
    test_data.to_csv(os.path.join(out_dir, 'test.csv'))


def read_npy(image_path):
    image = np.load(image_path)
    image = np.expand_dims(image, axis=0)
    image = sitk.GetImageFromArray(image)
    return image

if __name__ == '__main__':
    address = 'data/slicesFew/slice_metadataFew.csv'
    out_dir = 'data/slicesFew'
    train_valid_test_split(address, out_dir, train_ratio=0.6,
                           test_ratio=0.2, random_state=1)
    metadata_dile_path = address
    tsfm = forge.Compose([
        forge.Lambda(image_transformer=read_npy, mask_transformer=read_npy, p=1),
        forge.MinMaxScaler(),
        forge.ToNumpy()
    ])
    ds = IMGDataset(metadata_file_path=metadata_dile_path, transforms=tsfm)

    img, msk = ds[50]
    print('Images: ', img.shape, img.dtype, (img.min(), img.max()))
    print('Contour: ', msk.shape, msk.dtype, (msk.min(), msk.max()))
    plt.imshow(np.squeeze(img), cmap='gray')
    plt.show()
    plt.imshow(np.squeeze(msk), cmap='gray')
    plt.show()
