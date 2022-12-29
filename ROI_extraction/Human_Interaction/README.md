# Human Interaction part for UM ROI extraction

## Introduction
This code contains 2 tools: centroid based boundary decision; boundary refinement.

### Centroid based boundary decision
This tool helps to determine which centroids contain high-/mixed-/low-quality image regions. 10 random images are shown to users and users can simply click the images to select the high-quality iamge regions. The tool will output the statistics of each centroid (how many image regions are annotated as high-/low-quality)

Run
```Shell
python BoundaryDecision.py
```

### Boundary Refinement
This tool offers users interaction to refine the decision boundary in the feature space. The users can move, zoom in, zoom out the cytopathology image or the corresponding state image (high-/mixed-/low-quality states) to reannotate the image region. All image regions with similar image-level feature are also reannotated together. There exist some suspicious image regiosn shown on the state image (pink) to recommend users to reannotate them first.

Run
```Shell
python BoundaryRefinement.py
```

