import torch
from src.callbacks import CallbackHandler
from torch.utils.data import DataLoader
import os
from torch.cuda.amp import GradScaler, autocast
import random
 
 
class Project():
    '''
    A keras-like class to control your project.
    The most high level class of the project.
    Everything else should be changed in the dependencies.
    This class creates the training loop and handles the callbacks.
    Use callbacks to change the training loop, like changing the learning rate or saving the model.
    '''
 
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        
 
    def compile(self, discriminator_pixel, discriminator_people, generator_pixel, generator_people, pre_train=False, pre_train_path=None):
        '''
        Compiles the model and casts to device
 
        Args:
            model (torch.nn.Module): your neural network model
        '''
        self.discriminator_pixel = discriminator_pixel.to(self.device)
        self.discriminator_people = discriminator_people.to(self.device)
        self.generator_pixel = generator_pixel.to(self.device)
        self.generator_people = generator_people.to(self.device)

        if (pre_train):
            self.discriminator_pixel.load_state_dict(torch.load(os.path.join(pre_train_path,'discriminator_pixel.pth')))
            self.discriminator_people.load_state_dict(torch.load(os.path.join(pre_train_path,'discriminator_people.pth')))
            self.generator_pixel.load_state_dict(torch.load(os.path.join(pre_train_path,'generator_pixel.pth')))
            self.generator_people.load_state_dict(torch.load(os.path.join(pre_train_path,'generator_people.pth')))
 
    def predict(self, inputs):
        '''
        Method to predict the output of the model for a single input batch.
 
        Args:
            inputs (_type_): your input data
 
        Returns:
            (_type_): output of the model
        '''
 
        self.model.eval()
        with torch.no_grad():
            inputs = inputs.float().to(self.device)
            outputs = self.model(inputs)
        return outputs
 
    def fit(self,
            train_dataset,
            optimizer_generators=None,
            optimizer_discriminators=None,
            epochs=5,
            batch_size=1,
            shuffle=True,
            learning_rate_generators=0.001,
            learning_rate_discriminators=0.001,
            callbacks=[],
            loss_function_adversarial = None,
            loss_function_consistency = None,
            lambda_cycle = 10,
            lambda_identity = 5,
            buffer_size=20
            ):
        '''
        Method to train the model.
        Use this to change the training loop if needed.
        Notice that any change that is not structural should be done using callbacks.
        For example, if you want to change the learning rate during training, you should use a callback.
        Otherwise, if you are training for example a GAN, you should change the training loop.
 
        Args:
            train_dataset (Dataset): dataset for the training set.
            val_dataset (Dataset optional): dataset for the evaluation set. Defaults to None. If None, the training set is used.
            optimizer (optimizer, optional): optimizer used to train the model. Defaults to None.
            epochs (int, optional): number of epochs to train the model for. Defaults to 10.
            batch_size (int, optional): batch size for training. Defaults to 1.
            shuffle (bool, optional): whether to shuffle or not the data before training and testing. Defaults to True.
            learning_rate (float, optional): learning rate. Defaults to 0.001.
            callbacks (list of Callbacks, optional): callbacks to run during trainig. Defaults to [].
            loss (loss function, optional): loss function to use for training. Defaults to None. Not actually optional.
       
        Returns:
            dict: dictionary containing history of the training
        '''
 
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=shuffle, num_workers=4)
        self.scaler = GradScaler()
       
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer_generators = optimizer_generators
        self.optimizer_discriminators = optimizer_discriminators
        self.learning_rate_generators = learning_rate_generators
        self.learning_rate_discriminators = learning_rate_discriminators
        self.callbacks = callbacks
        self.loss_function_adversarial = loss_function_adversarial
        self.loss_function_consistency = loss_function_consistency
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        self.buffer_size=buffer_size
 
        for param_groups in self.optimizer_generators.param_groups:
            param_groups['lr'] = self.learning_rate_generators
        
        for param_groups in self.optimizer_discriminators.param_groups:
            param_groups['lr'] = self.learning_rate_discriminators
 
        self.train_loss_discriminator = []
        self.train_loss_generator = []
        self.epoch_loss_discriminator = torch.tensor(0.0, device=self.device)
        self.epoch_loss_generator = torch.tensor(0.0, device=self.device)
        self.stop_training = False

        self.buffer_discriminator_pixel = []
        self.buffer_discriminator_people = []
 
        callback_handler = CallbackHandler(callbacks, learner=self)
        callback_handler.on_train_begin()
 
        for self.epoch in range(1, epochs+1):
            callback_handler.on_epoch_begin()
            self.generator_people.train()
            self.generator_pixel.train()
            self.discriminator_people.train()
            self.discriminator_pixel.train()
           
            for self.batch, (people, pixel_arts) in enumerate(self.train_loader):
                callback_handler.on_batch_begin()

                people = people.to(self.device)
                pixel_arts = pixel_arts.to(self.device)
 
                # Training discriminators
                with autocast():
                    fake_people = self.generator_people(pixel_arts)
                    if len(self.buffer_discriminator_people) < self.buffer_size:
                        self.buffer_discriminator_people.append(fake_people.detach())
                    else:
                        index = random.randrange(0, self.buffer_size)
                        self.buffer_discriminator_people[index] = fake_people.detach()

                    random_fake_people = self.buffer_discriminator_people[random.randrange(0, len(self.buffer_discriminator_people))]
                    outputs_fake_people = self.discriminator_people(random_fake_people)
                    outputs_real_people = self.discriminator_people(people)

                    discriminator_people_real_loss = self.loss_function_adversarial(outputs_real_people, torch.ones_like(outputs_real_people))
                    discriminator_people_fake_loss = self.loss_function_adversarial(outputs_fake_people, torch.zeros_like(outputs_fake_people))

                    discriminator_people_loss = discriminator_people_real_loss + discriminator_people_fake_loss


                    fake_pixel = self.generator_pixel(people)
                    if len(self.buffer_discriminator_pixel) < self.buffer_size:
                        self.buffer_discriminator_pixel.append(fake_pixel.detach())
                    else:
                        index = random.randrange(0, self.buffer_size)
                        self.buffer_discriminator_pixel[index] = fake_pixel.detach()

                    random_fake_pixel = self.buffer_discriminator_pixel[random.randrange(0, len(self.buffer_discriminator_pixel))]
                    outputs_fake_pixel = self.discriminator_pixel(random_fake_pixel)
                    outputs_real_pixel = self.discriminator_pixel(pixel_arts)

                    discriminator_pixel_real_loss = self.loss_function_adversarial(outputs_real_pixel, torch.ones_like(outputs_real_pixel))
                    discriminator_pixel_fake_loss = self.loss_function_adversarial(outputs_fake_pixel, torch.zeros_like(outputs_fake_pixel))

                    discriminator_pixel_loss = discriminator_pixel_real_loss + discriminator_pixel_fake_loss

                    discriminator_loss = (discriminator_people_loss + discriminator_pixel_loss)/2

                self.optimizer_discriminators.zero_grad()
                self.scaler.scale(discriminator_loss).backward()
                self.scaler.step(self.optimizer_discriminators)
                self.scaler.update()

                # Training generators
                with autocast():
                    outputs_fake_people = self.discriminator_people(fake_people)
                    outputs_fake_pixel = self.discriminator_pixel(fake_pixel)

                    generator_people_loss = self.loss_function_adversarial(outputs_fake_people, torch.ones_like(outputs_fake_people))
                    generator_pixel_loss = self.loss_function_adversarial(outputs_fake_pixel, torch.ones_like(outputs_fake_pixel))


                    cycle_people = self.generator_people(fake_pixel)
                    cycle_pixel = self.generator_pixel(fake_people)

                    cycle_people_loss = self.loss_function_consistency(people, cycle_people)
                    cycle_pixel_loss = self.loss_function_consistency(pixel_arts, cycle_pixel)


                    identity_people = self.generator_people(people)
                    identity_pixel = self.generator_pixel(pixel_arts)

                    identity_people_loss = self.loss_function_consistency(people, identity_people)
                    identity_pixel_loss = self.loss_function_consistency(pixel_arts, identity_pixel)

                    generator_loss = (generator_people_loss 
                                    + generator_pixel_loss 
                                    + lambda_cycle*(cycle_people_loss + cycle_pixel_loss) 
                                    + lambda_identity*(identity_people_loss + identity_pixel_loss))

                self.optimizer_generators.zero_grad()
                self.scaler.scale(generator_loss).backward()
                self.scaler.step(self.optimizer_generators)
                self.scaler.update()

                self.epoch_loss_discriminator += discriminator_loss
                self.epoch_loss_generator += generator_loss
               
                callback_handler.on_batch_end()
           
            self.train_loss_discriminator.append((self.epoch_loss_discriminator).item()/len(self.train_loader))
            self.train_loss_generator.append((self.epoch_loss_generator).item()/len(self.train_loader))
 
            self.epoch_loss_discriminator = torch.tensor(0.0, device=self.device)
            self.epoch_loss_generator = torch.tensor(0.0, device=self.device)
           
            callback_handler.on_epoch_end()
 
            if (self.stop_training): break
 
        callback_handler.on_train_end()
        self.history = {'train_loss_discriminator': self.train_loss_discriminator,
                        'train_loss_generator': self.train_loss_generator}
 
        return self.history