import copy
import os
import torch
import gc
import torch.nn as nn

def dice(inputs, targets, smooth=1):
    with torch.no_grad():
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)
        inputs = (inputs > 0.5).type(torch.float32)
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (
                inputs.sum() + targets.sum() + smooth)

        return dice

class Trainer(object):
    def __init__(self, model, loaders, optimizer, criterion,
                 epochs, device, model_dir, logger):
        self.device = device
        self.model = torch.nn.DataParallel(model)
        self.model = model.to(self.device)
        self.loaders = loaders
        self.optimizer = optimizer 
        self.criterion = criterion
        self.epochs = epochs
        self.model_dir = model_dir
        self.logger = logger

    def fit(self):

        for epoch in range(self.epochs):
            # Train
            self.model.train()
            
            num_batches = 0
            epoch_loss = 0.0
            epoch_dice = 0.0
            print('')
            for batch in self.loaders['train']:
                images, contours = batch
                images = images.to(self.device)
                contours = contours.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, contours)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                with torch.no_grad():
                    epoch_dice = dice(outputs, contours)
                num_batches += 1
                del images
                del contours
                del outputs
                
            epoch_loss /= num_batches
            train_msg = ('Train ---> Epoch: {0}, Loss: {1:0.4f}, Dice score: '
                         '{2:0.4f} ')
            print(train_msg.format(epoch+1, epoch_loss, epoch_dice))
            self.logger.info(train_msg.format(epoch+1, epoch_loss, epoch_dice))
            
            # Evaluate. 
            eval_loss = self.performance(epoch, 'valid')
            
            # Assemble the model state.
            state = {
                    'epoch': epoch,
                    'loss': eval_loss,
                    'model': copy.deepcopy(self.model),
            }
            # Save the model 
            torch.save(state, os.path.join(self.model_dir,
                                           'Model_{:0>4}.pth'.format(epoch)))
                
    def performance(self, epoch, phase):
        self.model.eval()
        assert phase in {'test', 'valid'}
        valid_msg = ('Valid ---> Epoch: {0}, Loss: {1:0.4f}, Dice score: '
                     '{2:0.4f}')
        test_msg = ('Test ---> Loss: {0:0.4f}, Dice score: {1:0.4f}')
        with torch.no_grad(): 
            num_batches = 0
            epoch_loss = 0.0
            epoch_dice = 0.0
            for batch in self.loaders[phase]: 
                images, contours = batch 
                images = images.to(self.device)
                contours = contours.to(self.device)
                
                outputs = self.model(images)
                epoch_loss += self.criterion(outputs, contours).item()
                epoch_dice += dice(outputs, contours)

                num_batches += 1
                
            epoch_loss /= num_batches
            epoch_dice /= num_batches
            if phase == 'valid':
                msg = valid_msg.format(epoch+1, epoch_loss, epoch_dice)
            else:
                msg = test_msg.format(epoch_loss, epoch_dice)
            self.logger.info(msg)
            print(msg)
        return epoch_loss
