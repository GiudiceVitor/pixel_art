import torch
import torch.nn.functional as F
 
 
############## CHANGE THIS PART TO USE YOUR OWN LOSSES ##############
 
 
class MeanAbsoluteError(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, outputs, labels):
        '''
        Calculates the mean absolute error
 
        Args:
            outputs (_type_): predictions of the model
            labels (_type_): ground truth
 
        Returns:
            loss: torch tensor of the loss containing the loss and the gradients
        '''
 
        return F.l1_loss(outputs, labels)
 
class MeanSquaredEerror(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, outputs, labels):
        '''
        Calculates the mean squared error
 
        Args:
            outputs (_type_): predictions of the model
            labels (_type_): ground truth
 
        Returns:
            loss: torch tensor of the loss containing the loss and the gradients
        '''
 
        return F.mse_loss(outputs, labels)