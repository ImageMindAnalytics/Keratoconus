from lightning.pytorch.callbacks import Callback
import torchvision
import torch

import matplotlib.pyplot as plt
from neptune.types import File

class AutoEncoderImageLogger(Callback):
    def __init__(self, num_images=16, log_steps=100):
        self.log_steps = log_steps
        self.num_images = num_images
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % self.log_steps == 0:
            with torch.no_grad():
                img = batch[pl_module.hparams.img_column]
                seg = batch[pl_module.hparams.seg_column]

                noise_amount = torch.rand(img.shape[0]).to(pl_module.device) # Pick random noise amounts
                img_noisy = pl_module.corrupt(img, noise_amount)

                max_num_image = min(img.shape[0], self.num_images)
                grid_img = torchvision.utils.make_grid(img[0:max_num_image]).permute(1, 2, 0)
                
                trainer.logger.experiment['img'].upload(File.as_image(grid_img.cpu().numpy()))
                
                grid_img2 = torchvision.utils.make_grid(img_noisy[0:max_num_image]).permute(1, 2, 0).clamp(0, 1)
                trainer.logger.experiment['noisy'].upload(File.as_image(grid_img2.cpu().numpy()))

                grid_seg = torchvision.utils.make_grid(seg[0:max_num_image]).permute(1, 2, 0)/seg.max()
                trainer.logger.experiment['seg'].upload(File.as_image(grid_seg.cpu().numpy()))

                x_hat, z_mu, z_sigma = pl_module(img_noisy)

                grid_x_hat = torchvision.utils.make_grid(x_hat[0:max_num_image]).permute(1, 2, 0).clamp(0, 1)
                trainer.logger.experiment['x_hat'].upload(File.as_image(grid_x_hat.cpu().numpy()))