"""
Training loops for models.
"""
import time
import os
from collections import defaultdict

import numpy as np
import tqdm
import torch
import torchvision

from torch.utils import data as t_data
from torch.autograd import Variable

from ..utils import loss_functions
from ..utils import torch_helpers
from ..utils import helper

def save_model(model_results, model, model_trainer):
    model_train_state_dict = {
            'epoch': model_trainer.epoch, 
            'gen_optimizer': model_trainer.gen_optimizer,
            'discrim_optimizer': model_trainer.discrim_optimizer,
            'loss': None
            }

    model_results.save_model(model, model_train_state_dict)

class GanModelTrainer:
    def __init__(self, model, params, sample_dir=None, model_results=None):
        self.model = model
        self.generator, self.discriminator = model.generator, model.discriminator
        self.params = params
        self.sample_dir = sample_dir
        self.model_results = model_results

        self.results = defaultdict(list)

        # Set device
        self.device = torch.device(params.device) if torch.cuda.is_available() else torch.device('cpu')

        self.generator.to(self.device), self.discriminator.to(self.device)

        # Setup loss and optimizers
        self.discrim_optimizer = None

        if self.params.use_gan_loss:
            #self.gan_loss = loss_functions.gan_loss(params, self.device)
            self.gan_loss = loss_functions.select_gan_loss(self.params.gan_loss, self.params, self.device)
            self.discrim_optimizer = torch.optim.Adam(self.discriminator.parameters(), self.params.learning_rate)

        self.gen_optimizer = torch.optim.Adam(self.generator.parameters(), self.params.learning_rate)

        self.content_loss = loss_functions.select_content_loss(self.params.content_loss, self.params, self.device)

    def train(self, data, num_epochs, val_data=None):
        self.generator.train()
        self.discriminator.train()

        x_data_len_tr = int(len(data.low_res) * .875)

        x_low_data, x_high_data = data.low_res[:x_data_len_tr], data.high_res[:x_data_len_tr]
        x_low_data_val, x_high_data_val = data.low_res[x_data_len_tr:], data.high_res[x_data_len_tr:]

        print(f'Train Data: {x_data_len_tr}, Val Data: {len(x_low_data_val)}')

        train_data_loader = torch_helpers.create_data_loader(x_low_data, x_high_data, 
                self.params.batch_size, self.params.shuffle)

        print('Starting training...')
        for self.epoch in tqdm.tqdm(range(num_epochs)):
            start_time = time.time() 

            discrim_loss_total = 0.
            gen_loss_total = 0.
            train_pixel_err = 0.

            # === Training === 
            # Iterate over minibatches
            for batch_id, (x_low, x_high) in enumerate(train_data_loader):
                x_low, x_high = (
                        Variable(x_low).to(self.device),
                        Variable(x_high).to(self.device)
                )

                discrim_loss = torch.tensor(0.).to(self.device)
                gen_loss = torch.tensor(0.).to(self.device)

                # Compute GAN loss
                x_high_gen = self.generator(x_low)

                if self.params.use_gan_loss:
                    discrim_gan_loss, gen_gan_loss = self.gan_loss(x_high, x_high_gen, self.discriminator)

                    discrim_loss += discrim_gan_loss
                    gen_loss += self.params.gan_loss_scale * gen_gan_loss

                # Compute content loss
                if self.content_loss is not None:
                    gen_loss += self.params.content_loss_scale * self.content_loss(x_high_gen, x_high)

                # Compute error
                train_pixel_err += loss_functions.mse_loss()(x_high, x_high_gen).item()

                # Track total epoch loss for output
                discrim_loss_total += discrim_loss.item()
                gen_loss_total += gen_loss.item()

                # Backward pass on generator
                self.gen_optimizer.zero_grad()
                gen_loss.backward()
                self.gen_optimizer.step()

                # Backward pass on discriminator
                if self.params.use_gan_loss:
                    self.discrim_optimizer.zero_grad()
                    discrim_loss.backward(retain_graph=True)
                    self.discrim_optimizer.step()

            gen_loss_total /= batch_id
            discrim_loss_total /= batch_id
            train_pixel_err /= batch_id

            # === Output === 
            duration_sec = time.time() - start_time
            output_str = f'Epoch {self.epoch} ({duration_sec:.2f} seconds) - Train Loss: ({discrim_loss_total:.3f}, {gen_loss_total:.3f}), Train Err: {train_pixel_err:.3f}'

            # Compute "validation" loss
            val_gan_loss, val_pixel_err = self.compute_validation_loss(x_low_data_val, x_high_data_val)
            output_str += f', Val Loss: ({val_gan_loss[0]:.3f}, {val_gan_loss[1]:.3f}), Val Err: {val_pixel_err:.3f}'

            tqdm.tqdm.write(output_str)

            self.results['tr_pixel_err'].append(train_pixel_err)
            self.results['tr_gan_loss'].append((discrim_loss_total, gen_loss_total))

            self.results['val_pixel_err'].append(val_pixel_err)
            self.results['val_gan_loss'].append(val_gan_loss)

            if self.epoch % 10 == 0:
                save_model(self.model_results, self.model, self) 


            # Generate sample images
            if self.sample_dir:
                self.generate_sample_image(x_low_data[:3], x_high_data[:3], f'train_sample_{self.epoch}.png')
                self.generate_sample_image(x_low_data_val[:3], x_high_data_val[:3], f'val_sample_{self.epoch}.png')


    def compute_validation_loss(self, x_low_data_val, x_high_data_val):
        val_data_loader = t_data.DataLoader(torch_helpers.Dataset(x_low_data_val, x_high_data_val), 
                batch_size=self.params.batch_size, shuffle=self.params.shuffle)

        discrim_loss_total = 0
        gen_loss_total = 0
        train_pixel_err = 0

        with torch.no_grad():
            self.generator.eval()
            self.discriminator.eval()

            # Iterate over mini-batches
            for batch_id, (x_low, x_high) in enumerate(val_data_loader):
                x_low, x_high = (
                        Variable(x_low).to(self.device),
                        Variable(x_high).to(self.device)
                )

                x_high_gen = self.generator(x_low)
                if self.params.use_gan_loss:
                    discrim_loss, gen_loss = self.gan_loss(x_high, x_high_gen, self.discriminator)

                    gen_loss_total += gen_loss.item()
                    discrim_loss_total += discrim_loss.item()

                # Compute content loss
                if self.content_loss:
                    gen_loss_total += self.content_loss(x_high, x_high_gen)

                train_pixel_err += loss_functions.mse_loss()(x_high, x_high_gen).item()

            self.generator.train()
            self.discriminator.train()

            discrim_loss_total /= batch_id
            gen_loss_total /= batch_id
            train_pixel_err /= batch_id

        return (discrim_loss_total, gen_loss_total), train_pixel_err

    def generate_sample_image(self, x_low, x_high, name):
        x_low_dim = x_low.shape[1]
        x_high_dim = x_high.shape[1]

        with torch.no_grad():
            self.generator.eval()
            sample = self.generator(torch.tensor(x_low).float().view(-1,1,x_low_dim,x_low_dim).to(self.device))
            self.generator.train()

        output_path = os.path.join(self.sample_dir, name)

        pad_size = (x_high_dim - x_low_dim) // 2

        lo = torch.tensor([np.pad(arr, pad_size, 'constant') for arr in x_low]).float()
        hi = torch.tensor(x_high).float()
        o = sample.cpu().squeeze()

        out = torch.cat((lo, hi, o)).unsqueeze(1)

        torchvision.utils.save_image(out, output_path, nrow=3)
