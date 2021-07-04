import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import SimpleITK as sitk
import forge.src.transform as forge
from utils import read_image
from utils import read_metadata
from utils import ToTorchTensor

class Single3DImageDataset(Dataset):

    def __init__(self, path, transforms):
        super(Dataset, self).__init__()
        self.image = read_image(path)
        self.image_array = sitk.GetArrayFromImage(self.image)
        n = self.image.GetDepth()
        self.image_array = [(self.image_array[i]).copy() for i in range(n)]
        self.transforms = transforms

    def __getitem__(self, item):
        img = self.image_array[item]
        img = np.expand_dims(img, axis=0)
        img = sitk.GetImageFromArray(img)
        if self.transforms is not None:
            img, _ = self.transforms(image=img, mask=None)
        return img

    def __len__(self):
        return self.image.GetDepth()



class Segmentor(object):
    def __init__(self, configs, transforms, threshold=0.05):
        self.configs = configs
        self.threshold = threshold
        self.transforms = transforms
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(configs['pretrained_model_path'], map_location='cpu')
        model = checkpoint['model']
        self.model = torch.nn.DataParallel(model)
        self.model = model.to(self.device)

    def __call__(self, image_path, contour_path):
        ds = Single3DImageDataset(image_path, self.transforms)
        loader = DataLoader(dataset=ds,
                            batch_size=self.configs['batch_size'],
                            shuffle=False,
                            num_workers=self.configs['num_workers'])
        self.model.eval()
        with torch.no_grad():
            contour = []
            for i, images in enumerate(loader):
                images = images.to(self.device)
                outputs = self.model(images)
                outputs = outputs.cpu().numpy()
                for i, pred in enumerate(outputs):
                    pred = np.squeeze(pred)
                    msk = np.zeros_like(pred, dtype=np.uint8)
                    msk[pred >= self.threshold] = 1
                    contour.append(msk)
            contour = np.array(contour)
            contour = sitk.GetImageFromArray(contour)
            contour.CopyInformation(ds.image)
            print('Writing to ', contour_path)
            sitk.WriteImage(contour, contour_path)

if __name__ == '__main__':
    config_path = 'config.json'
    with open(config_path, 'r') as f:
        configs = json.load(f)
        configs['num_workers'] = 0
    # All files should be located inside root. They can be nested.
    experiment_dir = f'results/{configs["experiment"]}'
    root = 'data/LungDataset/'
    metadata_file_path = 'data/LungDataset/to_predict.csv'
    paths = read_metadata(metadata_file_path)
    output_dir = os.path.join(experiment_dir, 'test')
    models = {'Model_A': 'results/A2/train/Pretrained_Model.pth'}

    transforms = forge.Compose([
            forge.Resize((configs['width'], configs['height'], 1)),
            forge.UnitNormalize(p=1),
            ToTorchTensor(out_image_dtype=torch.float32,
                          out_mask_dtype=torch.float32)
        ])
    for model_name, model_path in models.items():
        configs['pretrained_model_path'] = model_path
        segmentor = Segmentor(configs, transforms=transforms, threshold=0.5)
        for item in paths:
            image_path = item['Image']
            contour_path = os.path.join(output_dir,
                                        model_name,
                                        image_path[len(root):])
            os.makedirs(os.path.dirname(contour_path), exist_ok=True)
            segmentor(image_path, contour_path)
            print(f'Contoured {image_path}')
