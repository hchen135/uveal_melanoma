All cytopathology images are processed by K-means clustering with average intensity values of 512x512 areas. The k-means clustering has 2 centroids and we select the one with low intensity. All 512x512 images belonging to this centroids are saved as the input of this code.

All images are saved to ``CONFIG.INPUT_DIR``/Slide \*/\*.png
The image name is ``$A$_TileLoc_$B$_$C$.png``. A is the tile ind, B, C means the top-left corner of the image is Bx512 and Cx512.

To run the code, please change the ``config*.py`` yo fit your own dataset.

Run
```Shell
python train.py --phase=
```
