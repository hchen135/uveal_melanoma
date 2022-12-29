# training example
python train.py --config=UMSLIC_config --save_folder=weights/11242020/ --save_interval=1000

# evaluation: image display
python eval.py --trained_model=weights/11242020/UMSLIC_79_4000.pth --score_threshold=0.15 --top_k=50 --config=UMSLIC_config --images=data/UMSLIC_val:data/UMSLIC_valresult


# evaluation: generate class 1 umap
python eval.py --trained_model=weights/11242020/UMSLIC_79_4000.pth --score_threshold=0.15 --top_k=50 --config=UMSLIC_Nature_umap_gene_config --output_path=results/11242020/umap_4000_circle_Nature/ --mapping_gene

# evaluation: generate class 2 umap
python eval.py --trained_model=weights/11242020/UMSLIC_59_3000.pth --score_threshold=0.15 --top_k=50 --config=UMSLIC_class2_umap_gene_config --output_path=results/11242020/umap_3000_circle_class2/ --mapping_gene

# evaluation: umap projection for specific slide
python eval.py --trained_model=weights/11182020/UMSLIC_79_4000.pth --score_threshold=0.15 --top_k=50 --config=UMSLIC_umap_proj_config --output_path=results/11182020/ --mapping_proj --proj_slide=51
