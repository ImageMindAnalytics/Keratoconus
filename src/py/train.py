import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch

import lightning as L

from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
# from pytorch_lightning.strategies.ddp import DDPStrategy
from lightning.pytorch.strategies import DDPStrategy

from lightning.pytorch.loggers import NeptuneLogger
# from pytorch_lightning.plugins import MixedPrecisionPlugin

import pickle

import SimpleITK as sitk

import dataset
import nets
import logger


def main(args):

    DM = getattr(dataset, args.data_module)
    data_module = DM(**vars(args))

    NN = getattr(nets, args.nn)    
    model = NN(**vars(args))

    callbacks = []

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=2,
        monitor='val_loss',
        save_last=True
        
    )

    callbacks.append(checkpoint_callback)

    if args.use_early_stopping:
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=args.patience, verbose=True, mode="min")
        callbacks.append(early_stop_callback)
    
    logger_neptune = None

    if args.neptune_tags:
        logger_neptune = NeptuneLogger(
            project='ImageMindAnalytics/Keratoconus',
            tags=args.neptune_tags,
            api_key=os.environ['NEPTUNE_API_TOKEN'],
            log_model_checkpoints=False
        )
        LOGGER = getattr(logger, args.logger)    
        image_logger = LOGGER(log_steps=args.log_steps)
        callbacks.append(image_logger)

    
    trainer = Trainer(
        logger=logger_neptune,
        log_every_n_steps=args.log_steps,
        max_epochs=args.epochs,
        max_steps=args.steps,
        callbacks=callbacks,
        accelerator='gpu', 
        devices=torch.cuda.device_count(),
        strategy=DDPStrategy(find_unused_parameters=True),
    )
    
    trainer.fit(model, datamodule=data_module, ckpt_path=args.model)


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='KC train')

    hparams_group = parser.add_argument_group('Hyperparameters')
    hparams_group.add_argument('--epochs', help='Max number of epochs', type=int, default=200)
    hparams_group.add_argument('--patience', help='Max number of patience for early stopping', type=int, default=20)
    hparams_group.add_argument('--steps', help='Max number of steps per epoch', type=int, default=-1)    

    input_group = parser.add_argument_group('Input')
    
    input_group.add_argument('--nn', help='Type of neural network', type=str, default="AutoEncoderKL")        
    input_group.add_argument('--model', help='Model to continue training', type=str, default= None)

    input_group.add_argument('--data_module', help='Data module type', required=True, type=str, default=None)

    
    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--out', help='Output directory', type=str, default="./")
    output_group.add_argument('--use_early_stopping', help='Use early stopping criteria', type=int, default=0)
    output_group.add_argument('--monitor', help='Additional metric to monitor to save checkpoints', type=str, default=None)
    
    log_group = parser.add_argument_group('Logging')
    log_group.add_argument('--neptune_tags', help='Neptune tags', type=str, nargs="+", default=None)
    log_group.add_argument('--logger', help='Neptune tags', type=str, default=None)
    log_group.add_argument('--log_steps', help='Log every N steps', type=int, default=5)

    initial_args, unknownargs = parser.parse_known_args()

    NN = getattr(nets, initial_args.nn)    
    NN.add_model_specific_args(parser)

    DM = getattr(dataset, initial_args.data_module)
    parser = DM.add_data_specific_args(parser)

    args = parser.parse_args()

    main(args)
