import os
import json
import logging
import torch
from torch import nn
from torch.utils.data import DataLoader

import SimpleITK as sitk
import forge.src.transform as forge
from utils import ToTorchTensor
from utils import read_npy
import dataset
import train


def runner(configs, action='train'):
    # Check output directory.
    # Check output directory.
    dir_address = os.path.join(configs['result_dir'],
                               configs['experiment'],
                               action)
    os.makedirs(dir_address, exist_ok=True)
    # Define logger.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=os.path.join(dir_address, 'experiment.log'),
                        filemode='a',
                        level=logging.DEBUG)
    logger = logging.getLogger(configs['experiment'])

    # Define data loaders.
    phases = ['train', 'valid', 'test']
    transforms = {
        'train': forge.Compose([
            forge.Lambda(image_transformer=read_npy, mask_transformer=read_npy,
                         p=1),
            forge.RandomRotation(angles=(0, 0, 30),
                                 interpolator=sitk.sitkLinear, p=0.90),
            forge.RandomCrop(size=(256, 256, 1), p=0.7),
            forge.RandomFlipX(p=0.5),
            forge.RandomFlipY(p=0.5),
            forge.BionomialBlur(repetition=1, p=0.5),
            forge.SaltPepperNoise(noise_prob=0.01, random_seed=None, p=0.5),
            forge.Resize((configs['width'], configs['height'], 1)),
            forge.UnitNormalize(p=1),
            ToTorchTensor(out_image_dtype=torch.float32, out_mask_dtype=torch.float32)
        ]),
        'valid': forge.Compose([
            forge.Lambda(image_transformer=read_npy, mask_transformer=read_npy,
                         p=1),
            forge.Resize((configs['width'], configs['height'], 1)),
            forge.UnitNormalize(p=1),
            ToTorchTensor(out_image_dtype=torch.float32,
                          out_mask_dtype=torch.float32)
        ]),
        'test': forge.Compose([
            forge.Lambda(image_transformer=read_npy, mask_transformer=read_npy,
                         p=1),
            forge.Resize((configs['width'], configs['height'], 1)),
            forge.UnitNormalize(p=1),
            ToTorchTensor(out_image_dtype=torch.float32,
                          out_mask_dtype=torch.float32)
        ])
    }
    datasets = {phase: dataset.IMGDataset(
                                metadata_file_path=configs[f'{phase}_path'],
                                transforms=transforms[phase])
                for phase in phases
    }
    shuffled = {'train': True, 'valid': False, 'test': False}
    loaders = {phase: DataLoader(dataset=datasets[phase],
                                 batch_size=configs['batch_size'],
                                 shuffle=shuffled[phase],
                                 num_workers=configs['num_workers'],
                                 collate_fn=dataset.IMGDataset.collate)
               for phase in phases
    }
    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Criteria
    criteria = nn.BCEWithLogitsLoss()
    # Define or load a model.
    if configs['pretrained'] is True:
        # load a model
        checkpoint = torch.load(configs['pretrained_model_path'], map_location='cpu')
        model = checkpoint['model']
    else:
        raise ValueError('You should define a model here.')
    # Define optimizer.
    optimizer = torch.optim.Adam(model.parameters(), **configs['optimizer'])
    # Define trainer.
    trainer = train.Trainer(model,
                            loaders,
                            optimizer,
                            criteria,
                            configs['epochs'],
                            device,
                            dir_address,
                            logger)
    if action == 'train':
        trainer.fit()
    else:
        trainer.performance(0, phase='test')


def main():
    config_path = 'config.json'
    training = True
    # Load configs.
    assert os.path.exists(config_path), 'config file does not exist'
    try:
        with open(config_path, 'r') as f:
            configs = json.load(f)
    except:
        print('Invalid config file')
        raise
    if training is True:
        print('The model will be trained')
        runner(configs)
    else:
        print('The model will be tested')
        runner(configs, action='test')

if __name__ == '__main__':
    main()