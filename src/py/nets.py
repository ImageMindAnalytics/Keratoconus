import lightning as L
from lightning.pytorch.core import LightningModule

from generative.losses.adversarial_loss import PatchAdversarialLoss
from generative.losses.perceptual import PerceptualLoss
from generative.networks import nets

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

from monai import transforms


class GaussianNoise(nn.Module):    
    def __init__(self, mean=0.0, std=0.05):
        super(GaussianNoise, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
    def forward(self, x):
        if self.training:
            return x + torch.normal(mean=self.mean, std=self.std, size=x.size(), device=x.device)
        return x
    
class RandCoarseShuffle(nn.Module):    
    def __init__(self, prob=0.75, holes=4, spatial_size=8):
        super(RandCoarseShuffle, self).__init__()
        self.t = transforms.RandCoarseShuffle(prob=prob, holes=holes, spatial_size=spatial_size)
    def forward(self, x):
        if self.training:
            return self.t(x)
        return x

class SaltAndPepper(nn.Module):    
    def __init__(self, prob=0.05):
        super(SaltAndPepper, self).__init__()
        self.prob = prob
    def __call__(self, x):
        noise_tensor = torch.rand(x.shape)
        salt = torch.max(x)
        pepper = torch.min(x)
        x[noise_tensor < self.prob/2] = salt
        x[noise_tensor > 1-self.prob/2] = pepper
        return x

class AutoEncoderKL(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.autoencoderkl = nets.AutoencoderKL(
            spatial_dims=2,
            in_channels=self.hparams.in_channels,
            out_channels=self.hparams.out_channels,
            num_channels=(128, 256, 384),
            latent_channels=self.hparams.latent_channels,
            num_res_blocks=1,
            norm_num_groups=32,
            attention_levels=(False, False, True),
        )

        self.perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="alex")

        self.automatic_optimization = False

        self.discriminator = nets.PatchDiscriminator(spatial_dims=2, num_layers_d=3, num_channels=64, in_channels=self.hparams.in_channels, out_channels=self.hparams.out_channels)                

        self.adversarial_loss = PatchAdversarialLoss(criterion="least_squares")

    def corrupt(self, x, amount):
        """Corrupt the input `x` by mixing it with noise according to `amount`"""
        noise = torch.rand_like(x)
        amount = amount.view(-1, 1, 1, 1) # Sort shape so broadcasting works
        return x*(1-amount) + noise*amount 
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        hparams_group = parent_parser.add_argument_group("AutoencoderKL")

        hparams_group.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='Learning rate')
        hparams_group.add_argument('--weight_decay', help='Weight decay for optimizer', type=float, default=0.01)

        hparams_group.add_argument('--perceptual_weight', help='Perceptual weight', type=float, default=0.001)
        hparams_group.add_argument('--adversarial_weight', help='Adversarial weight', type=float, default=0.01)        
        hparams_group.add_argument('--kl_weight', help='Weight decay for optimizer', type=float, default=1e-6)    
        hparams_group.add_argument('--autoencoder_warm_up_n_epochs', help='Warmup epochs', type=float, default=10)                
        hparams_group.add_argument('--in_channels', help='Number of input channels in the image', type=int, default=3)    
        hparams_group.add_argument('--out_channels', help='Number of output channels in the image', type=int, default=3)    
        hparams_group.add_argument('--latent_channels', help='Latent Channels', type=int, default=3)
        
        # Encoder parameters

        return parent_parser

    def configure_optimizers(self):
        optimizer_g = optim.AdamW(self.autoencoderkl.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)

        optimizer_d = optim.AdamW(self.discriminator.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)

        return [optimizer_g, optimizer_d]
    
    def compute_loss_generator(self, x, reconstruction, z_mu, z_sigma):
        recons_loss = F.l1_loss(reconstruction.float(), x.float())
        p_loss = self.perceptual_loss(reconstruction.float(), x.float())
        kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        loss_g = recons_loss + (self.hparams.kl_weight * kl_loss) + (self.hparams.perceptual_weight * p_loss)

        if self.trainer.current_epoch >= self.hparams.autoencoder_warm_up_n_epochs:
            logits_fake = self.discriminator(reconstruction.contiguous().float())[-1]
            generator_adv_loss = self.adversarial_loss(logits_fake, target_is_real=True, for_discriminator=False)
            loss_g += self.hparams.adversarial_weight * generator_adv_loss        

        return loss_g, recons_loss
    
    def compute_loss_discriminator(self, x, reconstruction):
        logits_fake = self.discriminator(reconstruction.contiguous().detach())[-1]
        loss_d_fake = self.adversarial_loss(logits_fake, target_is_real=False, for_discriminator=True)
        logits_real = self.discriminator(x.contiguous().detach())[-1]
        loss_d_real = self.adversarial_loss(logits_real, target_is_real=True, for_discriminator=True)
        return (loss_d_fake + loss_d_real) * 0.5


    def training_step(self, train_batch, batch_idx):
        x = train_batch[self.hparams.img_column]

        optimizer_g, optimizer_d = self.optimizers()

        noise_amount = torch.rand(x.shape[0]).to(self.device) # Pick random noise amounts
        x = self.corrupt(x, noise_amount)

        reconstruction, z_mu, z_sigma = self(x)

        loss_g, recons_loss = self.compute_loss_generator(x, reconstruction, z_mu, z_sigma)

        optimizer_g.zero_grad()
        self.manual_backward(loss_g)
        optimizer_g.step()
        self.untoggle_optimizer(optimizer_g)
        
        loss_d = 0.0
        if self.trainer.current_epoch >= self.hparams.autoencoder_warm_up_n_epochs:
            loss_d = self.compute_loss_discriminator(x, reconstruction)

            optimizer_d.zero_grad()
            self.manual_backward(loss_d)
            optimizer_d.step()
            self.untoggle_optimizer(optimizer_d)

        self.log("train_loss_recons", recons_loss)
        self.log("train_loss_g", loss_g)
        self.log("train_loss_d", loss_d)

        return {"train_loss_g": loss_g, "train_loss_d": loss_d}
        

    def validation_step(self, val_batch, batch_idx):
        x = val_batch[self.hparams.img_column]

        reconstruction, z_mu, z_sigma = self(x)
        recon_loss = F.l1_loss(x.float(), reconstruction.float())

        self.log("val_loss", recon_loss, sync_dist=True)

    def forward(self, images):        
        return self.autoencoderkl(images)