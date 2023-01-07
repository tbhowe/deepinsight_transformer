#%%
import torch
from transformers import ViTModel, ViTImageProcessor, ViTConfig
import os
import time
import torch.nn.functional as F
import timm


class DeepInsightVitModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.model_path='vit-base-patch16-224-in21k'
        self.layers = timm.create_model('vit_base_patch32_224_in21k', pretrained=True, num_classes=0)
        for param in self.layers.parameters():
            param.grad_required = False  
        self.regression_head_1=torch.nn.Linear(in_features= 1536, out_features= 1)
        self.regression_head_2=torch.nn.Linear(in_features= 1536, out_features= 1)
        

    def forward(self, x):
        output=self.layers(x)
        out1 = self.regression_head_1(output)
        out2 = self.regression_head_2(output)
        return out1,out2
    
    @staticmethod
    def combined_loss(out1, out2, labels):
        loss_fn1 = F.mse_loss()
        loss_fn2 = F.mse_loss()
        loss1 = loss_fn1(out1, labels[0])
        loss2 = loss_fn2(out2, labels[1])
        return loss1 + loss2
        
if __name__ == "__main__":
    model=DeepInsightVitModel()
    model.parameters()

# %%
