import os
import csv

import torch
import numpy as np
import SimpleITK as sitk



def read_npy(image_path):
    image = np.load(image_path)
    image = np.expand_dims(image, axis=0)
    image = sitk.GetImageFromArray(image)
    return image


def read_metadata(metadata_file_path):
    with open(metadata_file_path, 'r') as fin:
        data = []
        num_records = None
        for row in csv.DictReader(fin):
            if len(row) == 0:
                continue
            if num_records is None:
                num_records = len(row)
            else:
                assert len(row) == num_records
            data.append(row)
    return data


def read_DICOM_CT_from_dir(dir_path):
    """ Read a CT image (or its contours) from a directory.
    Args:
        dir_path (str): Address of the directory containing DICOM files.
    Returns: A SimpleITK.Image.
    """
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dir_path)
    if len(series_ids) == 0:
        raise ValueError('No DICOM file in directory:\n{}'.format(dir_path))
    slice_paths = reader.GetGDCMSeriesFileNames(dir_path, series_ids[0])
    reader.SetFileNames(slice_paths)
    image = reader.Execute()
    # Reading matadata from one slice, currently SimpleITK does not support
    # extracting metadata when reading a series
    reader = sitk.ImageFileReader()
    reader.SetFileName(slice_paths[0])
    img = reader.Execute()
    for k in img.GetMetaDataKeys():
        value = img.GetMetaData(k)
        image.SetMetaData(k, value)
    return image

def read_image(path):
    if os.path.isdir(path):
        return read_DICOM_CT_from_dir(path)
    elif os.path.isfile(path):
        image = sitk.ReadImage(path)
    return image


class ToTorchTensor(object):
    """Convert an image and a mask to Torch Tensor.
    Input image and mask must be Numpy ndarray or SimpleITK Image.

    Args:
        out_image_dtype (Torch dtype): Assign a new Torch data type to the output image.
            The default value is `None`. This means the output image data type
            remains unchanged.
        out_mask_dtype (Torch dtype): Assign a new Torch data type to the output
            mask-image. The default value is `None`. This means the output mask-image
            data type remains unchanged.
    """
    def __init__(self, out_image_dtype=None, out_mask_dtype=None):
        self.img_dtype = out_image_dtype
        self.msk_dtype = out_mask_dtype

    def __call__(self, image, mask=None, *args, **kwargs):
        # Convert image and mask.
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
            if mask is not None:
                mask = torch.from_numpy(mask)
        elif isinstance(image, sitk.SimpleITK.Image):
            image = torch.from_numpy(sitk.GetArrayFromImage(image))
            if mask is not None:
                mask = torch.from_numpy(sitk.GetArrayFromImage(mask))
        else:
            raise ValueError(
                'image and mask must be in type torch Tensor or SimpleITK Image.')
        # Change image and mask dtypes.
        if self.img_dtype is not None:
            image = image.type(self.img_dtype)
        if mask is not None and self.msk_dtype is not None:
            mask = mask.type(self.msk_dtype)
        return image, mask

    def __repr__(self):
        msg = '{} (out_image_dtype={}, out_mask_dtype={})'
        return msg.format(self.__class__.__name__,
                          self.img_dtype,
                          self.msk_dtype)

