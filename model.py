#%%
import torch
from transformers import ViTModel, ViTImageProcessor, ViTConfig
import os
import time

class DeepInsightVitModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.coniguration= ViTConfig()
        self.model_path='google/vit-base-patch16-224-in21k'
        self.layers = ViTModel.from_pretrained(self.model_path)
        for param in self.layers.parameters():
            param.grad_required = False  
        self.regression_head_1=torch.nn.Linear(in_features= 768, out_features= 1)
        self.regression_head_2=torch.nn.Linear(in_features= 768, out_features= 1)
        
    def forward(self, x):
        output=self.layers(x)
        pooler_output=output.pooler_output
        out1 = self.head1(pooler_output)
        out2 = self.head2(pooler_output)
        return out1,out2
        
if __name__ == "__main__":
    model=DeepInsightVitModel()


# %%
