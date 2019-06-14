"""
Training loops for models.
"""
import time
import os

import tqdm
import torch
import torchvision

from torch.utils import data as t_data
from torch.autograd import Variable

from ..utils import loss_functions
from ..utils import torch_helpers

class GanModelTrainer:
    def __init__(self, model, params, sample_dir=None):
        self.generator, self.discriminator = model.generator, model.discriminator
        self.params = params
        self.sample_dir = sample_dir

        # Set device
        self.device = torch.device(params.device) if torch.cuda.is_available() else torch.device('cpu')

        # Setup loss and optimizers
        self.discrim_optimizer = None

        if self.params.use_gan_loss:
            self.gan_loss = loss_functions.gan_loss(params)
            self.discrim_optimizer = torch.optim.Adam(self.generator.parameters(), self.params.learning_rate)

        self.gen_optimizer = torch.optim.Adam(self.discriminator.parameters(), self.params.learning_rate)

        self.content_loss = loss_functions.select_content_loss(self.params.content_loss, self.params)

    def train(self, data, num_epochs, val_data=None):
        self.generator.train()
        self.discriminator.train()

        x_low_data, x_high_data = data.low_res, data.high_res

        train_data_loader = torch_helpers.create_data_loader(x_low_data, x_high_data, 
                self.params.batch_size, self.params.shuffle)

        print('Starting training...')
        for self.epoch in tqdm.tqdm(range(num_epochs)):
            start_time = time.time() 

            discrim_loss_total = torch.tensor(0.).to(self.device)
            gen_loss_total = torch.tensor(0.).to(self.device)

            train_pixel_err = torch.tensor(0.).to(self.device)

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
                    gen_loss += gen_gan_loss

                # Compute content loss
                gen_loss += self.params.content_loss_scale * self.content_loss(x_high_gen, x_high)

                # Track total epoch loss for output
                discrim_loss_total += discrim_loss
                gen_loss_total += gen_loss

                # Backward pass on discriminator
                if self.params.use_gan_loss:
                    self.discrim_optimizer.zero_grad()
                    discrim_loss.backward(retain_graph=True)
                    self.discrim_optimizer.step()

                # Backward pass on generator
                self.gen_optimizer.zero_grad()
                gen_loss.backward(retain_graph=True)
                self.gen_optimizer.step()

                # Compute error
                train_pixel_err += loss_functions.mse_loss()(x_high, x_high_gen)

            # === Output === 
            duration_sec = time.time() - start_time
            output_str = f'Epoch {self.epoch} ({duration_sec:.2f} seconds) - Train Loss: ({discrim_loss.item():.2f}, {gen_loss.item():.2f}), Train Err: {train_pixel_err.item():.2f}'

            # Compute "validation" loss
            if val_data:
                val_loss, val_pixel_err = self.compute_validation_loss(val_data)
                output_str += f', Val Loss: {val_loss}, Val Err: {val_pixel_err}'

            tqdm.tqdm.write(output_str)

            # Generate sample images
            if self.sample_dir:
                real_sample_path = os.path.join(self.sample_dir, 'train_sample_real.png')
                if not os.path.exists(real_sample_path):
                    torchvision.utils.save_image(torch.tensor(x_high_data[0]).float(), real_sample_path)

                self.generate_sample_image(x_low_data[0], f'train_sample_{self.epoch}.png')

                if val_data:
                    real_sample_path_val = os.path.join(self.sample_dir, 'val_sample_real.png')
                    if not os.path.exists(real_sample_path_val):
                        torchvision.utils.save_image(torch.tensor(val_data.high_res[0]).float(), real_sample_path_val)

                    self.generate_sample_image(val_data.low_res[0], f'val_sample_{self.epoch}.png')


    def compute_validation_loss(self, val_data):
        x_low_val_data, x_high_val_data = val_data.low_res, val_data.high_res

        val_data_loader = t_data.DataLoader(torch_helpers.Dataset(x_low_val_data, x_high_val_data), 
                batch_size=self.params.batch_size, shuffle=self.params.shuffle)

        discrim_loss_total = torch.tensor(0.).to(self.device)
        gen_loss_total = torch.tensor(0.).to(self.device)

        train_pixel_err = torch.tensor(0.).to(self.device)

        with torch.no_grad():
            self.generator.eval()
            self.discriminator.eval()

            # Iterate over mini-batches
            for x_low, x_high in val_data_loader:
                x_high_gen = self.generator(x_low)
                discrim_loss, gen_loss = self.loss(x_high_gen, x_high, self.discriminator)

                discrim_loss_total += discrim_loss
                gen_loss_total += gen_loss

                train_pixel_err += loss_functions.mse_loss()(x_high, x_high_gen)

            self.generator.train()
            self.discriminator.train()

        return (discrim_loss_total, gen_loss_total), train_pixel_err

    def generate_sample_image(self, x_low, name):
        x_low_dim = x_low.shape[0]
        with torch.no_grad():
            self.generator.eval()
            sample = self.generator(torch.tensor(x_low).float().view(1,1,x_low_dim,x_low_dim))
            self.generator.train()

        output_path = os.path.join(self.sample_dir, name)
        torchvision.utils.save_image(sample, output_path)
