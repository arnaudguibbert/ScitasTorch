import logging

from time import perf_counter

import torch 
import torch.nn as nn

logger = logging.getLogger(__name__)


class TrainModel():

    def __init__(self,model:nn.Module,
                 optimizer:torch.optim,
                 criterion:callable,
                 train_data:torch.utils.data,
                 epochs:int,
                 time_limit:float,
                 lr_scheduler:torch.optim.lr_scheduler=None,
                 valid_data:torch.utils.data=None,
                 **kwargs) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_data = train_data
        self.valid_data = valid_data
        self.epochs = epochs
        self.time_limit = time_limit
        self.lr_scheduler = lr_scheduler
        self.current_epoch = 0
        self.last_epoch = None

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        

    def load_checkpoint(self,checkpoint:dict):
        if checkpoint is None:
            return None
        else:
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            if self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            self.current_epoch = checkpoint["epoch"]


    def export_checkpoint(self,path:str):
        torch.save({'epoch': self.current_epoch,
                   'model_state_dict': self.model.state_dict(),
                   'optimizer_state_dict': self.optimizer.state_dict(),
                   'lr_scheduler': self.lr_scheduler.state_dict() 
                                   if self.lr_scheduler is not None else None},
                    path)


    def train(self) -> bool:

        limit_reached = False
        start = perf_counter()

        logger.info("Training started")

        self.model.to(self.device)
        self.criterion.to(self.device)


        self.model.train()

        for epoch in range(self.current_epoch,self.epochs):
            train_loss = 0

            for (feat, label) in self.train_data:
                feat, label = feat.to(self.device), label.to(self.device)
                output = self.model(feat)
                loss = self.criterion(output,label)
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
                with torch.no_grad():
                    train_loss += loss

            logger.info("Epoch [%s/%s] - Training Loss : %s",
                        epoch,self.epochs,
                        train_loss.item() / self.train_data.batch_size)
            
            if self.valid_data is not None:
                valid_loss = 0
                for (feat, label) in self.valid_data:
                    feat, label = feat.to(self.device), label.to(self.device)
                    output = self.model(feat)
                    with torch.no_grad():
                        valid_loss += self.criterion(output,label)
                
                logger.info("Epoch [%s/%s] - validation Loss : %s",
                            epoch,self.epochs,
                            train_loss.item() / self.valid_data.batch_size)

            timer = perf_counter() - start
            if timer > self.time_limit:
                limit_reached = True
                self.last_epoch = epoch + 1
                break

        return limit_reached

        





        