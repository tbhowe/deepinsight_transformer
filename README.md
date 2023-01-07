# deepinsight_transformer
Adaptation of the basic idea behind DeepInsight by Markus Frey and Caswell Barry, but instead of a basic convnet, use
transfer learning of generalised image models, specifically leveraging transformers. 

Based on https://github.com/CYHSM/DeepInsight/

## Milestone 1: Create a Dataset class
 Class takes in example wavelet-transformed data and returns timebins (Freq x Channels arrays PIL images).
 __getitem__ method also transforms PIL image to features using VitImageProcessor from Huggingface Transformers

 ## Milestone 2: Create the Model

 ## Milestone 3: Define the train loop

 ## Mileston 4: Iterative improvements of the model on the example dataset

 TODO - compare transformers to basic ResNet + FC approach
 TODO - K-fold x-vals
 TODO - Test whether Maxpool layer or Features gives better results
 TODO - experiment with FC layers complexity
 TODO - condsider simple CNN or RNN over output features
 TODO - alternative transformer model architectures
 

 ## Milestone 5 - build out and generalise model

 TODO - variable numbers and types of data
 TODO - binary and multiclass classification 
 TODO - consider case of NREM and REM sleep-associated replay - how would the model handle time-compressed replay in SWRs? How about real-time replay in REM as per Louie & Wilson 2001?