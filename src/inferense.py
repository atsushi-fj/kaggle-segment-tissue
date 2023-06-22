import torch
from tqdm.auto import tqdm


def eval_model(model,
               data_loader,
               loss_fn,
               device):
    """Returns a dictionary containing the results of model predicting on data_loader."""
    loss = 0
    model.eval()
    with torch.inference_mode():
        for images, maksks in tqdm(data_loader):
            images, masks = images.to(device), maksks.to(device)
            outputs = model(images)

            loss += loss_fn(outputs, masks)
            
        loss /= len(data_loader)

    return {"model_name": model.__class__.__name__, 
            "model_loss": loss.item()}
    