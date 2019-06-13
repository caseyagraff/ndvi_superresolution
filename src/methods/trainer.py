"""
Training loops for models.
"""
import time

import tqdm
import torch
import torchvision

from torch.utils import data as t_data

from ..utils import loss_functions
from ..utils import torch_helpers

class GanModelTrainer:
    def __init__(self, model, params, sample_dir=None):
        self.generator, self.discriminator = model.generator, model.discriminator
        self.params = params
        self.sample_dir = sample_dir

        # Set device
        self.device = torch.device(params.device) if torch.cuda.is_available() else torch.device('cpu')

        # Setup optimizer
        self.discrim_optimizer = torch.optim.Adam(self.generator.parameters(), self.params.learning_rate)
        self.gen_optimizer = torch.optim.Adam(self.discriminator.parameters(), self.params.learning_rate)

        # Setup loss
        self.gan_loss = model.gan_loss
        self.content_loss = loss_functions.select_content_loss(self.params.content_loss, self.params)

    def train(self, data, num_epochs, val_data=None):
        self.generator.train()
        self.discriminator.train()

        x_low_data, x_high_data = data

        train_data_loader = torch_helpers.create_data_loader(x_low_data, x_high_data, 
                self.params.batch_size, self.params.shuffle)

        for epoch in tqdm.tqdm(range(num_epochs)):
            start_time = time.time() 

            discrim_loss_total = torch.tensor(0.).to(self.device)
            gen_loss_total = torch.tensor(0.).to(self.device)

            train_pixel_error = torch.tensor(0.).to(self.device)

            # === Training === 
            # Iterate over minibatches
            for batch_id, (x_low, x_high) in enumerate(train_data_loader):
                x_low, x_high = x_low.to(device), x_high.to(device)

                # Compute GAN loss
                x_high_gen = self.generator(x_low)
                discrim_loss, gen_loss = self.gan_loss(x_high_gen, x_high, self.discriminator)

                # Compute content loss
                gen_loss += self.params.content_loss_scale * self.content_loss(x_high_gen, x_high)

                discrim_loss_total += discrim_loss
                gen_loss_total += gen_loss

                # Backward pass on discriminator
                self.discrim_optimizer.zero_grad()
                discrim_loss.backward(retain_graph=True)
                self.discrim_optimizer.step()

                # Backward pass on generator
                self.gen_optimizer.zero_grad()
                gen_loss.backward(retain_graph=True)
                self.gen_optimizer.step()

                # Compute error
                train_pixel_error += loss_functions.l2_pixel_error(x_high, x_high_gen)

            # === Output === 
            duration_sec = time.time() - start_time
            output_str = f'Epoch {epoch} ({duration_sec} seconds) - Train Loss: {train_loss}, \
                    Train Err: {train_pixel_err}'

            # Compute "validation" loss
            if val_data:
                val_loss, val_pixel_err = self.compute_validation_loss(val_data)
                output_str += f', Val Loss: {val_loss}, Val Err: {val_pixel_err}'

            print(output_str)

            # Generate sample images
            if self.sample_dir:
                self.generate_sample_image(x_low_data[0], f'train_sample_{epoch}.png')
                self.generate_sample_image(x_low_val_data[0], f'val_sample_{epoch}.png')

    def compute_validation_loss(self, val_data):
        x_low_val_data, x_high_val_data = val_data

        val_data_loader = t_data.DataLoader(torch_helpers.Dataset(x_low_val_data, x_high_val_data), 
                batch_size=self.params.batch_size, shuffle=self.params.shuffle)

        discrim_loss_total = torch.tensor(0.).to(self.device)
        gen_loss_total = torch.tensor(0.).to(self.device)

        train_pixel_error = torch.tensor(0.).to(self.device)

        with torch.no_grad():
            self.generator.eval()
            self.discriminator.eval()

            # Iterate over mini-batches
            for x_low, x_high in val_data_loader:
                x_high_gen = self.generator(x_low)
                discrim_loss, gen_loss = self.loss(x_high_gen, x_high, self.discriminator)

                discrim_loss_total += discrim_loss
                gen_loss_total += gen_loss

                train_pixel_error += loss_functions.l2_pixel_error(x_high, x_high_gen)

            self.generator.train()
            self.discriminator.train()

        return (discrim_loss_total, gen_loss_total), train_pixel_error

    def generate_sample_image(self, x_low, name):
        with torch.no_grad():
            self.generator.eval()
            sample = self.generator(x_low)
            self.generator.train()

        output_path = os.path.join(self.sample_dir, name)
        torchvision.utils.save_image(sample_train_img, output_path)


