# Rule-based learning for Uveal Melanoma GEP classification

## Introduction

This code uses the cell-distribution in the pie chart to analyze and predict the GEP classification for each slide. We partition the pie chart into 12 parts. Then we use the portion of cells in each partition and the ratio of the portion of cells between each 2 partitions as the input variables (totally 78 input variables). This code includes the GEP classification with svm, logistic regression, ANN and rule-set. As the same as the paper, we also have ensemble option to enlarge the training space.

### Data

The input data is the output of the ``Cell_Segmentation`` pipeline. We use the ``umap_proj_*.json`` as the inpouts. Please change the ``--umap_info_dir`` argument in bash files to target the correct input folder.

### Training
The main training file is ``main_UM.py``. We use bash files to call the training file to run.

When trying to run logistic regression model without ensemble:
```Shell
sh script_logistic.sh
```

When trying to run logistic regression model with ensemble:
```Shell
sh script_logistic_ensemble.sh
```

When trying to run svm model without ensemble:
```Shell
sh script_svm.sh
```

When trying to run svm model with ensemble:
```Shell
sh script_svm_ensemble.sh
```

When trying to run ann model without ensemble:
```Shell
sh script_ann.sh
```

When trying to run ann model with ensemble:
```Shell
sh script_ann_ensemble.sh
```

When trying to run ruleset model without ensemble:
```Shell
sh script_ruleset.sh
```

When trying to run ruleset model with ensemble:
```Shell
sh script_ruleset_ensemble.sh
```
