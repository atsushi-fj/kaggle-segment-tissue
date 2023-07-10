import argparse
from pycocotools import _mask as coco_mask 
from ultralytics import YOLO

from dataset import COCODataset
from src.utils import load_config, create_display_name

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
    
    coco = COCODataset(
        annotations_filepath="../input/data/polygons.jsonl",
        images_dirpath="input/data/train"
    )
    
    coco(train_size=0.80, classes=[0, 1, 2])
    
    model = YOLO("yolov8x-seg.pt")
    
    model.train(
        # Project
        project=cfg.project,
        name=name,

        # Random Seed parameters
        deterministic=True,
        seed=cfg.seed,

        # Data & model parameters
        data="./dataset/coco.yaml", 
        save=True,
        save_period=5,
        pretrained=True,
        imgsz=512,

        # Training parameters
        epochs=cfg.epochs,
        batch=cfg.batch_size,
        workers=8,
        val=True,
        device=0,

        # Optimization parameters
        lr0=0.018,
        patience=3,
        optimizer="SGD",
        momentum=0.947,
        weight_decay=0.0005,
        close_mosaic=3,
    )
