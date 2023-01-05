# For which project:

This is the code for the paper: Direct Gene Expression Profile Prediction for Uveal Melanoma from Digital Cytopathology Images via Deep Learning and Salient Image Region Identification.

## Introduction
This method first use attention mechanism to train ROIs with slide-level annotation. Then, the deep features are weighted averaged into ROI-level features by the attention weights. Furthermore, ROI-level features are weighted averaged into slide-level features by the attention weights. Finally, the slide-level features are trained with slide-level annotations by ANN.

## Inputs
The inputs are the same as the ``ROI_extraction`` project.

## Training
```Shell
python train.py --dataset=UM
```


