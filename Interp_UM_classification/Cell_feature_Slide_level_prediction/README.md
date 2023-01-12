# Slide level prediction with cell level features

## Introduction

This code uses the cell features from ``Cell_Segmentation`` to further predict slide-level survival. For each slide, there are `K` cells extracted. Each cell has its own outputed cell-level feature. This algorithm use ANN to encode the features into a length`8` feature. Then another linear+ReLU learns the attention weights for this cell. Finally, all of the length `8` cell-features in one slide is weighted averaged by the attention weights, so that for each slide, it has a length `8` slide-level feature. Finally, this feature are further learned to output the survival status by a linear+ReLU layer. Details can be found in ``model.py``.

### Data

The data are from the ``Cell_Segmentation`` with feature outputs.

### Preparation

We first get the data split in ``split`` fodler
```Shell
python split.py
```

### Training and testing
In default, in the training, we randomly choose 1000 cells for each slide to train. In the testing, we use all cells to get the results. In order to change the number of cells used in the training, please change the ``max_cella`` in ``main.py``.

```Shell
python main.py
```
