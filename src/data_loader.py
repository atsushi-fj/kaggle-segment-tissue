import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import json
import numpy as np
from skimage.draw import polygon2mask
from sklearn.model_selection import train_test_split
import gc


class CustomDataset(Dataset):
    def __init__(self, image_dir, labels_file, transform=None):
        # Open the labels file and load the JSON data
        with open(labels_file, 'r') as json_file:
            # Read each line in the JSON file and parse it as JSON
            self.json_labels = [json.loads(line) for line in json_file]

        # Set the image directory, labels file, and transformation
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.json_labels)

    def __getitem__(self, idx):
        # Load image
        image_path = os.path.join(self.image_dir, f"{self.json_labels[idx]['id']}.tif")
        image = Image.open(image_path)

        # Initialize mask
        mask = np.zeros((512, 512), dtype=np.float32)

        # Process annotations
        for annot in self.json_labels[idx]['annotations']:
            cords = annot['coordinates']
            if annot['type'] == "blood_vessel":
                # Iterate over the coordinates of the annotation
                for cord in cords:
                    # Extract the x and y coordinates from the coordinates list
                    rr, cc = np.array([i[1] for i in cord]), np.asarray([i[0] for i in cord])
                    # Set the corresponding pixels in the mask to 1
                    mask[rr, cc] = 1

        # Convert PIL Image and mask to PyTorch tensor
        image = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1)  # Shape: [C, H, W]
        mask = torch.tensor(mask, dtype=torch.float32)

        if self.transform:
            # Apply the transformation to the image if provided
            image = self.transform(image)

        return image, mask
    
    


def create_dataloader(train_dataset, val_dataset,
                      batch_size=32,
                      num_workers=os.cpu_count(),
                      pin_memory=True,
                      train_drop_last=True):
    
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory,
                                  drop_last=train_drop_last)
    if val_dataset is None:
        return train_dataloader
    
    val_dataloader = DataLoader(val_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 pin_memory=pin_memory)
    
    return train_dataloader, val_dataloader


def data_split(X, y,
               test_size=0.2,
               random_state=42,
               stratify=None):
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, 
                                                    test_size=test_size,
                                                    random_state=random_state,
                                                    stratify=stratify)
    del X, y
    gc.collect()
    train_dataset = CustomDataset(X_train, y_train)
    val_dataset = CustomDataset(X_val, y_val)
    
    return train_dataset, val_dataset

