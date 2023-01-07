#%%
from di_dataset import DeepInsightDataset
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
        self.model=timm.create_model('vit_base_patch32_224_in21k', pretrained=True, num_classes=0)
        model_config=self.model.pretrained_cfg
        resolved_config=timm.data.resolve_data_config(model_config)
        self.transform=timm.data.create_transform(**resolved_config)
        self.layers = self.model
        for param in self.layers.parameters():
            param.grad_required = False  
        self.regression_head_1=torch.nn.Linear(in_features= 768, out_features= 1)
        self.regression_head_2=torch.nn.Linear(in_features= 768, out_features= 1)
        

    def forward(self, x):
        output=self.layers(x)
        out1 = self.regression_head_1(output)
        out2 = self.regression_head_2(output)
        return out1,out2
    
    @staticmethod
    def combined_loss(out1, out2, labels):
        loss1 = F.mse_loss(out1,labels[0])
        loss2 = F.mse_loss(out2, labels[1])
        return loss1,loss2
        
if __name__ == "__main__":
    
    
    model=DeepInsightVitModel()
    my_dataset=DeepInsightDataset(transform=model.transform)
    test_features,test_labels=my_dataset.__getitem__(10)
    print(test_features)
    
    

    # feature_transform_test=model.transform(test_features)
    # print(feature_transform_test)
    # optimiser=torch.optim.AdamW
    # lr=0.0001
    # optimiser = optimiser(model.parameters(), lr=lr, weight_decay=0.001)
# %%
