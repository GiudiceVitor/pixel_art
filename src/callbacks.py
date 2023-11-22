import torch
from timeit import default_timer as timer
import os
import matplotlib.pyplot as plt
import torchvision.transforms as T
import numpy as np
 
class Callback():
    '''
    Base class for callbacks to extend.
    '''
 
    def __init__(self): pass
    def on_train_begin(self, learner): pass
    def on_train_end(self, learner): pass
    def on_epoch_begin(self, learner): pass
    def on_epoch_end(self, learner): pass
    def on_batch_begin(self, learner): pass
    def on_batch_end(self, learner): pass
    def on_loss_begin(self, learner): pass
    def on_loss_end(self, learner): pass
    def on_step_begin(self, learner): pass
    def on_step_end(self, learner): pass
 
 
class CallbackHandler():
    def __init__(self, callbacks, learner):
       
        '''
        A class to handle callbacks.
 
        Args:
            callbacks (list): list of callbacks to handle
            learner (Project): learner to pass to callbacks to access and modify its attributes
        '''
        self.callbacks = callbacks
        self.learner = learner
 
    def on_train_begin(self):
        for callback in self.callbacks:
            callback.on_train_begin(self.learner)
 
    def on_train_end(self):
        for callback in self.callbacks:
            callback.on_train_end(self.learner)
 
    def on_epoch_begin(self):
        for callback in self.callbacks:
            callback.on_epoch_begin(self.learner)
 
    def on_epoch_end(self):
        for callback in self.callbacks:
            callback.on_epoch_end(self.learner)
 
    def on_batch_begin(self):
        for callback in self.callbacks:
            callback.on_batch_begin(self.learner)
 
    def on_batch_end(self):
        for callback in self.callbacks:
            callback.on_batch_end(self.learner)
 
############## CHANGE THIS PART TO USE YOUR OWN CALLBACKS ##############

 
class ReduceLROnPlateau(Callback):
    def __init__(self, patience=1, min_lr=0, verbose=1, factor=0.1, min_delta=1e-4):
        '''
        A callback to reduce the learning rate when the validation loss has stopped improving.
 
        Args:
            patience (int, optional): number of epochs to wait to see improvement. Defaults to 1.
            min_lr (int, optional): mininum learning rate to reduce to. Defaults to 0.
            verbose (int, optional): whether or not to see visual feedback of the callback. Defaults to 1.
                                     verbose == 0: don't show anything
                                     verbose == 1: show a message when the learning rate is reduced
                                     verbose == 2: for debugging purposes
            factor (float, optional): factor to multiply the learning rate by. Defaults to 0.1.
            min_delta (_type_, optional): minimum difference between best metric and current metric to consider improvement. Defaults to 1e-4.
        '''
 
        super().__init__()
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose
        self.factor = factor
        self.min_delta = min_delta
        self.best_loss = 1e10
        self.wait = 0
 
    def on_epoch_end(self, learner):
        if(self. verbose == 2):
            print(f'loss: {learner.val_loss[-1]}')
            print(f'best_loss: {self.best_loss}')
 
        if self.best_loss - learner.val_loss[-1] < self.min_delta:
            self.wait += 1
        else:
            self.wait = 0
            if (self.verbose == 2): print(f'\n[ReduceLROnPlateau] wait = 0 because diff is {self.best_loss - learner.val_loss[-1]} and min_delta is {self.min_delta}')
            self.best_loss = learner.val_loss[-1]
           
        if self.wait >= self.patience:
            new_lr = max(learner.optimizer.param_groups[0]['lr'] * self.factor, self.min_lr)
            learner.optimizer.param_groups[0]['lr'] = new_lr
            learner.learning_rate = new_lr
            self.wait = 0
            if (self.verbose == 1) or (self.verbose == 2):
                print('\nReducing learning rate to {}\n'.format(new_lr))
        else:
            if (self.verbose == 2): print(f'\n[ReduceLROnPlateau] not reducing learning rate because wait is {self.wait} and patience is {self.patience}')
 
 
class ModelCheckpoint(Callback):
    def __init__(self, save_path=os.getcwd(), model_name='model.pth'):
        '''
        A callback to save the model.
 
        Args:
            save_best_only (bool, optional): whether or not to save only the best model. Defaults to True.
            save_path (String optional): path to save the model. Defaults to the current directory.
        '''
 
        super().__init__()
        self.save_path = save_path
        self.model_name=model_name
 
    def on_train_end(self, learner):
        os.mkdir(os.path.join(self.save_path, self.model_name))
        torch.save(learner.discriminator_pixel.state_dict(), os.path.join(self.save_path, self.model_name, 'discriminator_pixel.pth'))
        torch.save(learner.discriminator_people.state_dict(), os.path.join(self.save_path, self.model_name, 'discriminator_people.pth'))
        torch.save(learner.generator_pixel.state_dict(), os.path.join(self.save_path, self.model_name, 'generator_pixel.pth'))
        torch.save(learner.generator_people.state_dict(), os.path.join(self.save_path, self.model_name, 'generator_people.pth'))
 
 
class EarlyStopping(Callback):
    def __init__(self, patience=1, min_delta=1e-4, verbose=1):
        '''
        A callback to stop training when a metric has stopped improving.
 
        Args:
            patience (int, optional): number of epochs to wait to see improvement. Defaults to 1.
            min_delta (int, optional): minimum difference between best metric and current metric to consider improvement. Defaults to 1e-4.
            verbose (int, optional): whether or not to see visual feedback of the callback. Defaults to 1.
                                     verbose == 0: don't show anything;
                                     verbose == 1: show a message when the learning rate is reduced;
                                     verbose == 2: for debugging purposes.
        '''
 
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.wait = 0
        self.best_loss = 1e10
        self.best_model = None
 
    def on_epoch_end(self, learner):
        if self.best_loss - learner.val_loss[-1] < self.min_delta:
            self.wait += 1
        else:
            self.wait = 0
            if (self.verbose == 2): print(f'\n[EarlyStopping] wait = 0 because diff is {self.best_loss - learner.val_loss[-1]} and min_delta is {self.min_delta}\n')
            self.best_loss = learner.val_loss[-1]
 
        if self.wait >= self.patience:
            learner.stop_training = True
            if (self.verbose == 1) or (self.verbose == 2):
                learner.stop_training = True
                print('\nEarly stopping\n')
        else:
            if (self.verbose == 2): print(f'\n[EarlyStopping] not stopping training because wait is {self.wait} and patience is {self.patience}\n')
 
 
class Logger(Callback):
    def __init__(self):
        '''
        A callback to print metrics.
        '''
        self.elapsed_time = 0
        self.elapsed_epoch_time = 0
        self.batch_start_time = 0
        self.batch_end_time = 0
        super().__init__()
 
    def on_train_begin(self, learner):
        gpus = int(os.environ.get('WORLD_SIZE', 1))
        print(f'Running on {gpus} GPUs', end = '. ')
        print(f'Training on {len(learner.train_loader.dataset)} samples')
        self.initial_time = timer()
 
    def on_batch_begin(self, learner):
        self.batch_start_time = timer()
 
    def on_batch_end(self, learner):
        self.elapsed_time = timer() - self.initial_time
        self.elapsed_epoch_time = timer() - self.initial_time
        print(f'Epoch {learner.epoch}/{learner.epochs} >>> Batch {learner.batch+1}/{len(learner.train_loader)} -- Loss generator: {learner.epoch_loss_generator/(learner.batch+1):.5f},  Loss discriminator: {learner.epoch_loss_discriminator/(learner.batch+1):.5f},  Elapsed Time: {self.elapsed_epoch_time:.1f}', end='\r')
   
    def on_epoch_end(self, learner):
        self.elapsed_time = timer() - self.initial_time
        self.elapsed_epoch_time = timer() - self.initial_time
        print(f'Epoch {learner.epoch}/{learner.epochs} >>> Batch {learner.batch+1}/{len(learner.train_loader)} -- Loss generator: {learner.train_loss_generator[-1]:.5f},  Loss discriminator: {learner.train_loss_discriminator[-1]:.5f},  Elapsed Time: {self.elapsed_epoch_time:.1f}')
        self.elapsed_epoch_time = 0
   
    def on_train_end(self, learner):
        print(f'Finished training. Total elapsed time: {self.elapsed_time:.1f} seconds')
 

class GeneratorPlotteronNotebook(Callback):
    def __init__(self, plot_every=1):
        super().__init__()
        self.plot_every = plot_every

    def on_epoch_end(self, learner):
        if learner.epoch % self.plot_every == 0:
            learner.generator_people.eval()
            learner.generator_pixel.eval()
            with torch.no_grad():
                random_index = torch.randint(0, len(learner.train_loader.dataset), (1,))
                people, pixel_arts = learner.train_loader.dataset[random_index]
                people = people.unsqueeze(0).to(learner.device)
                pixel_arts = pixel_arts.unsqueeze(0).to(learner.device)
                fake_people = learner.generator_people(pixel_arts)
                fake_pixel = learner.generator_pixel(people)

                self.plot_images(pixel_arts, fake_people, people, fake_pixel, learner.device)

    def plot_images(self, pixel_arts, fake_people, people, fake_pixel, device):
        plt.figure(figsize=(8, 8))
        plt.subplot(2, 2, 1)
        self.imshow(pixel_arts.cpu().squeeze(0), title="Original Pixel Art", normalize=True)
        plt.subplot(2, 2, 2)
        self.imshow(fake_people.cpu().squeeze(0), title="Generated People", normalize=True)
        plt.subplot(2, 2, 3)
        self.imshow(people.cpu().squeeze(0), title="Original People", normalize=True)
        plt.subplot(2, 2, 4)
        self.imshow(fake_pixel.cpu().squeeze(0), title="Generated Pixel Art", normalize=True)
        plt.show()

    def imshow(self, img, title="", normalize=False):
        # Assuming the image is normalized, unnormalize it for display

        def scale_1_to_0(x):
            return (x + 1) / 2

        # Add it to your transformations
        transf = T.Compose([
            # ... (other transformations here, if needed)
            T.Lambda(scale_1_to_0)
        ])

        if normalize:
            img = transf(img)
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.title(title)
        plt.axis('off')
                

class LearningRateScheduler(Callback):
    def __init__(self, verbose=1, initial_epoch=5):
        '''
        A callback to schedule the learning rate.
 
        Args:
            schedule (function): function that takes the epoch and returns the learning rate
            verbose (int, optional): whether or not to see visual feedback of the callback. Defaults to 1.
                                     verbose == 0: don't show anything
                                     verbose == 1: show a message when the learning rate is reduced
                                     verbose == 2: for debugging purposes
        '''
 
        super().__init__()
        self.verbose = verbose
        self.initial_epoch = initial_epoch

    def on_train_begin(self, learner):
        self.initial_lr = learner.optimizer_discriminators.param_groups[0]['lr']
 
    def on_epoch_end(self, learner):
        if learner.epoch < self.initial_epoch:
            return
        lr_decay = (learner.epoch - 1) / (learner.epochs - 1)
        new_lr = self.initial_lr * (1 - lr_decay)
        
        for param_group in learner.optimizer_discriminators.param_groups:
            param_group['lr'] = new_lr
        for param_group in learner.optimizer_generators.param_groups:
            param_group['lr'] = new_lr

        if self.verbose > 0:
            print(f'\nLearning rate set to {new_lr}\n')
                
 
            