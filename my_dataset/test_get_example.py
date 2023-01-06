#%%
import pandas as pd
from transformers import ViTFeatureExtractor
import numpy as np
import torch
from PIL import Image

data=pd.read_pickle('data.pickle')

def get_example(data,idx):
    '''get one example from dataset'''
    feature_extractor=ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    image = data['features'].iloc[idx]
    image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
    image =np.transpose(image).swapaxes(1,2)
    image = Image.fromarray(image, 'RGB')
    pos_x = data['pos_x'].iloc[idx]
    pos_y =data['pos_y'].iloc[idx]
    input=feature_extractor(images=image)
    example={}
    example['pixel_values'] = input['pixel_values']
    example['labels'] = [pos_x,pos_y]
    return example

example=get_example(data,1)
print(example)
# %%
