# Cell Segmentation the UM interpretable classification

## Introduction
Given super-pixel annotated images, we train a instance level cell segmentation network with [YOLACT](https://github.com/dbolya/yolact). We slightly modified the original code and are shown in this repository.

### Preparation
This code only accepts pixel-level annotations.

To run this code in a completely different dataset, first, please prepare the data paths in ``data/config.py``. 
1. ``UM_class`` is defined in ``UMSLIC_CLASSES`` and ``UMSLIC_LABEL_MAP``.
2. Prepare the dataset metafile. Examples are in ``UMSLIC_dataset``.
3. Prepare the config of the modified model. Examples are in ``UMSLIC_config``.

### Data generation
To create the input for ``Cell_Segmentation`` pipeline, please first use ``Cell_Annotation`` to create the anntations, then follow ``network_input_sample/Slide 36/2459_TileLoc_153_58_0_1.npy`` to create the inputs.

Once got the ``npy`` files, use ``network_input_sample/split.py`` to create ``train/val`` set. Then use ``network_input_sample/coco_style_anno_generation.py`` to generate the networ input.

Example data generated is shown in [here](https://drive.google.com/file/d/1yvQiGth3-OAq0jovorvj8N4koBewYN8x/view?usp=sharing).

When using ``umap_gene`` or ``umap_proj`` in ``eval.py``, please refer to ``network_input_sample/coco_style_info_gene.py`` to create the network input.

### Training
Run ``train.py`` with the defined config file.
```Shell
python train.py --config=UMSLIC_config --save_folder=weights/11242020/ --save_interval=1000
```

### Evalutation - image display
Choose the pretrained model, and define the input and output path of the image folder. An pre-trained model which is used in the paper is at [here (`UMSLIC_59_3000.pth`)](https://drive.google.com/drive/folders/1j08BDQgGrdPG8LfFu5n-IyJK6nQfue0H?usp=sharing).
```Shell
python eval.py --trained_model=weights/11242020/UMSLIC_79_4000.pth --score_threshold=0.15 --top_k=50 --config=UMSLIC_config --images=data/UMSLIC_val:data/UMSLIC_valresult
```

We also use umap to cluster the extracted cells. You need to defined a new data config file for the umap generation to select which slides are used to generate the umap. The umap gene will always distort the projection into a unit circle.
### Evaluation - generate class 1 umap 
```Shell
python eval.py --trained_model=weights/11242020/UMSLIC_79_4000.pth --score_threshold=0.15 --top_k=50 --config=UMSLIC_Nature_umap_gene_config --output_path=results/11242020/umap_4000_circle_Nature/ --mapping_gene
```

After created the umap projection, we can use the same projection to create cell composition of new slides.
### Evaluation: umap projection for specific slide
```Shell
python eval.py --trained_model=weights/11182020/UMSLIC_79_4000.pth --score_threshold=0.15 --top_k=50 --config=UMSLIC_umap_proj_config --output_path=results/11182020/ --mapping_proj --proj_slide=51
```

When generating the umap and do projection for new slides, you can also store the corresponding cell features:
```Shell
python eval.py --trained_model=weights/11242020/CCSLIC_79_4000.pth --score_threshold=0.15 --top_k=50 --config=UMSLIC_umap_gene_config --output_path=results/11242020/umap_4000_circle_feature_store/ --mapping_gene --phi_rescale --feature_store
```
```Shell
python eval.py --trained_model=weights/11242020/UMSLIC_79_4000.pth --score_threshold=0.15 --top_k=50 --config=UMSLIC_umap_proj_config --output_path=results/11242020/umap_4000_feature_store/   --proj_slide=1  --feature_store;
```


