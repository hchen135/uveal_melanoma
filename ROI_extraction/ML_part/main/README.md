# UM-ROI-extraction

### Introduction
This is the code to use deep cluster network to cluster 256x256 image regions into cerntroids. The input image is 512x512. We use [BagNet](https://github.com/wielandbrendel/bag-of-local-features-models) to ensure that the 9x9 pixels in the output corresponds to 256x256 image region per pixel.

### Steps
All cytopathology images are processed by K-means clustering with average intensity values of 512x512 areas. The k-means clustering has 2 centroids and we select the one with low intensity. All 512x512 images belonging to this centroids are saved as the input of this code.

All images are saved to ``CONFIG.input_dir``/Slide \*/\*.png
The image name is ``$A$_TileLoc_$B$_$C$.png``. A is the tile ind, B, C means the top-left corner of the image is Bx512 and Cx512.

To run the code, please change the ``config*.py`` yo fit your own dataset.

Run experiment with BagNet with ctf (coarse to fine)
```Shell
python train.py --phase=bagnet_coarse_to_fine --resume=False
```
The corresponding config file is ``config_bagnet_ctf.py``

Run experiment with BagNet without ctf
```Shell
python train.py --phase=bagnet --resume=False
```
The corresponding config file is ``config_bagnet.py``

If adding std (standard deviation to reassign/insert centroids), change ``CONFIG.reassign_by_std`` to ``True`` or ``False``.
