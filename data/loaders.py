import logging

import torchvision
from torch.utils.data import DataLoader

def SetDataLoader(dataset_name:str,
                  root:str,
                  batch_size=32,**kwargs) -> DataLoader:
    
    if dataset_name == "MNIST":
        dataset = torchvision.datasets.MNIST(root,
                                             transform=torchvision.transforms.ToTensor(),**kwargs)
    else:
        logging.warn("Dataset requested not available : %s",dataset_name)
        return None
        
    loader = DataLoader(dataset,batch_size,num_workers=4,pin_memory=True)
    return loader