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
from engine import train
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
        
        train_image_folder = "../input/data/train"
        val_image_folder = "../input/data/test"
        labels_file = "../input/data/polygons.jsonl"
        
        train_dataset = CustomDataset(image_dir=train_image_folder,
                                      labels_file=labels_file)
        
        val_dataset = CustomDataset(image_dir=val_image_folder,
                                    labels_file=labels_file)
        
        train_dataloader, val_dataloader = create_dataloader(train_dataset, val_dataset,
                                                            batch_size=cfg.batch_size,
                                                            pin_memory=True,
                                                            train_drop_last=True)
        
        model = SegmentationModel()
        
        # Training model
        model.to(device)
        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        earlystopping = EarlyStopping(patience=cfg.patience, verbose=True)
        
        
        train(model, train_dataloader, val_dataloader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochs=cfg.epochs,
            earlystopping=earlystopping,
            model_name=cfg.model_path,
            device=device)
        
        model.load_state_dict(torch.load(f=cfg.load_model_path))
        model.to(device)
        
        result = eval_model(model=model,
                            data_loader=val_dataloader,
                            loss_fn=loss_fn,
                            device=device)
        
        print(f"\n{result}")