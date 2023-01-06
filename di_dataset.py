#%%
import os
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
from datasets import GeneratorBasedBuilder, DatasetBuilder
# from datasets import DatasetInfo , Features,  Value, Image
import pandas as pd
from transformers import ViTImageProcessor
import numpy as np

class DeepInsightDataset:
    '''dataset for DeepInsight with transformers, in the form of a pytorch dataset'''

    def __init__(self):
        self.data=pd.read_pickle('data.pickle')
        
    def __len__(self):
        return self.data.shape[0]

    def __repr__(self):
        return "hello"  # str(self.cities)

    
    def __getitem__(self, idx):
        '''get one example from dataframe
            
            inputs: dataframe, idx_to_get
        
            
            '''
        
        img = self.data['features'].iloc[idx]
        img=self.array_to_RGB(img)
        pos_x = self.data['pos_x'].iloc[idx]
        pos_y =self.data['pos_y'].iloc[idx]
        labels=[pos_x,pos_y]
        return img, labels

    @staticmethod
    def array_to_RGB(image):
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2) # make 3-channel
        image =np.transpose(image).swapaxes(1,2) # re-order axes to chan:x:y
        image = Image.fromarray(image, 'RGB')
        return(image)


my_dataset=DeepInsightDataset()
my_dataset.__getitem__(10)
# %%
