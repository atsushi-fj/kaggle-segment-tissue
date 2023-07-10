from glob import glob
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import wandb
import argparse
import torch 
from pathlib import Path
from torch.utils.data import DataLoader

from utils import load_config, create_display_name, seed_everything, EarlyStopping
from data_loader import CustomDataset, create_dataloader
from models.baseline.baseline import SegmentationModel
from models.baseline.unet import UNet
from engine import train, train_for_submit
from inferense import eval_model


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="segment model : kaggle segment tissue")
    
    parser.add_argument("-config", type=str, default="config1.yaml",
                        help="Set config file")
    
    args = parser.parse_args()
    
    cfg = load_config(file=args.config)
    name = create_display_name(experiment_name=cfg["experiment_name"],
                               model_name=cfg["model_name"],
                               extra=cfg["extra"])

    with wandb.init(project=cfg["project"],
                    name=name,
                    config=cfg):
        
        cfg = wandb.config
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {device} device")
        
        seed_everything(seed=cfg.seed)
        
        if cfg.mode == "train":
        
            train_dataset = CustomDataset(image_dir=cfg.train_image_folder,
                                        labels_file=cfg.train_labels_file)
            
            val_dataset = CustomDataset(image_dir=cfg.train_image_folder,
                                        labels_file=cfg.val_labels_file)
        
            train_dataloader, val_dataloader = create_dataloader(train_dataset, val_dataset,
                                                                batch_size=cfg.batch_size,
                                                                pin_memory=True,
                                                                train_drop_last=True)
        else:
            train_dataset = CustomDataset(image_dir=cfg.train_image_folder,
                                        labels_file=cfg.labels_file)
            
            train_dataloader = create_dataloader(train_dataset, None,
                                                batch_size=cfg.batch_size,
                                                pin_memory=True,
                                                train_drop_last=True)
        
        model = UNet(encoder_name=cfg.encoder_name,
                     encoder_weights=cfg.encoder_weights,
                     in_channels=cfg.in_channels,
                     classes=cfg.n_classes,
                     activation=cfg.activation)
        
        # Training model
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        earlystopping = EarlyStopping(patience=cfg.patience, verbose=True)
        
        if cfg.mode == "train":
            train(model, train_dataloader, val_dataloader,
                optimizer=optimizer,
                epochs=cfg.epochs,
                earlystopping=earlystopping,
                model_name=cfg.model_path,
                device=device)
        else:
            train_for_submit(model, train_dataloader,
                             optimizer=optimizer,
                             epochs=cfg.epochs,
                             model_name=cfg.model_path,
                             device=device)
        
        model.load_state_dict(torch.load(f=cfg.load_model_path))
        model.to(device)
        
        result = eval_model(model=model,
                            data_loader=val_dataloader,
                            device=device)
        
        print(f"\n{result}")
