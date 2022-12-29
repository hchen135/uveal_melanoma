import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
import torch.nn.functional as F
import cv2
import numpy as np
from .config import cfg
from pycocotools import mask as maskUtils
import random
import json

def get_label_map():
    if cfg.dataset.label_map is None:
        return {x+1: x+1 for x in range(len(cfg.dataset.class_names))}
    else:
        return cfg.dataset.label_map 

class COCOAnnotationTransform(object):
    """Transforms a COCO annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    """
    def __init__(self):
        self.label_map = get_label_map()

    def __call__(self, target, width, height):
        """
        Args:
            target (dict): COCO target json annotation as a python dict
            height (int): height
            width (int): width
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class idx]
        """
        scale = np.array([width, height, width, height])
        res = []
        for obj in target:
            if 'bbox' in obj:
                bbox = obj['bbox']
                label_idx = obj['category_id']
                if label_idx >= 0:
                    label_idx = self.label_map[label_idx] - 1
                final_box = list(np.array([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])/scale)
                final_box.append(label_idx)
                res += [final_box]  # [xmin, ymin, xmax, ymax, label_idx]
            else:
                print("No bbox found for object ", obj)

        return res


class COCODetection(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        set_name (string): Name of the specific set of COCO images.
        transform (callable, optional): A function/transform that augments the
                                        raw images`
        target_transform (callable, optional): A function/transform that takes
        in the target (bbox) and transforms it.
        prep_crowds (bool): Whether or not to prepare crowds for the evaluation step.
    """

    def __init__(self, image_path, info_file, transform=None,
                 target_transform=None,
                 dataset_name='MS COCO', has_gt=True, proj_slide=None):
        # Do this here because we have too many things named COCO
        from pycocotools.coco import COCO
        
        if target_transform is None:
            target_transform = COCOAnnotationTransform()

        self.root = image_path
        self.coco = COCO(info_file)
        
        self.proj_slide = proj_slide
        if proj_slide:
            if 'CC' in cfg.name:
                self.ids = [i for i in list(self.coco.imgs.keys()) if '_'.join(self.coco.imgs[i]['file_name'].split('/')[-2].split(' ')[0].split('_')[:2]) == proj_slide]
            else:
                self.ids = [i for i in list(self.coco.imgs.keys()) if self.coco.imgs[i]['file_name'].split('/')[-2].split(' ')[-1] == proj_slide]
            print('total number of projection imgs: ',len(self.ids))
        else:
            self.ids = list(self.coco.imgs.keys())
        
        self.transform = transform
        self.target_transform = COCOAnnotationTransform()
        
        self.name = dataset_name
        self.has_gt = has_gt

        if ('UM' in cfg.name or 'CC' in cfg.name) and cfg.apply_classification:
            self.UM_classification = True
            with open(cfg.class_info_path) as a:
                class_info = json.load(a)
                if 'UM' in cfg.name:
                    class_mapping_dict = {0:0,1:0,2:1}
                elif 'CC' in cfg.name:
                    class_mapping_dict = {0:1,1:1,2:0,3:1}
                self.UM_class_dict = {i:class_mapping_dict[class_info[i]['class_ind']] for i in class_info}
        else:
            self.UM_classification = False

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, (target, masks, num_crowds)).
                   target is the object returned by ``coco.loadAnns``.
        """
        im, gt, masks, h, w, num_crowds, semantic_annotated_area, UM_class = self.pull_item(index)
        return im, (gt, masks, num_crowds, semantic_annotated_area, UM_class)

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target, masks, height, width, crowd).
                   target is the object returned by ``coco.loadAnns``.
            Note that if no crowd annotations exist, crowd will be None
        """
        img_id = self.ids[index]

        if self.has_gt:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)

            # Target has {'segmentation', 'area', iscrowd', 'image_id', 'bbox', 'category_id'}
            if len(ann_ids) > 0:
                target = [x for x in self.coco.loadAnns(ann_ids) if x['image_id'] == img_id]
            else:
                target = []
        else:
            target = []
        # Separate out crowd annotations. These are annotations that signify a large crowd of
        # objects of said class, where there is no annotation for each individual object. Both
        # during testing and training, consider these crowds as neutral.
        crowd  = [x for x in target if     ('iscrowd' in x and x['iscrowd'])]
        target = [x for x in target if not ('iscrowd' in x and x['iscrowd'])]
        num_crowds = len(crowd)

        for x in crowd:
            x['category_id'] = -1

        # This is so we ensure that all crowd annotations are at the end of the array
        target += crowd
        # The split here is to have compatibility with both COCO2014 and 2017 annotations.
        # In 2014, images have the pattern COCO_{train/val}2014_%012d.jpg, while in 2017 it's %012d.jpg.
        # Our script downloads the images as %012d.jpg so convert accordingly.
        file_name = self.coco.loadImgs(img_id)[0]['file_name']
        if self.has_gt:
            semantic_seg_ind_dict = self.coco.loadImgs(img_id)[0]['semantic_seg_anno']
        if file_name.startswith('COCO'):
            file_name = file_name.split('_')[-1]
  
        path = osp.join(self.root, file_name)
        assert osp.exists(path), 'Image path does not exist: {}'.format(path)
        img = cv2.imread(path)
        height, width, _ = img.shape
        if self.has_gt:
            # get semantic_seg annotated area, only needed for UMSLIC
            semantic_annotated_area = self.semantic_annotated_area_generation(semantic_seg_ind_dict, height, width)
        if len(target) > 0:
            # Pool all the masks for this image into one [num_objects,height,width] matrix
            # COCO
            #masks = [self.coco.annToMask(obj).reshape(-1) for obj in target]
            # UMSLIC
            masks = [self.UMSLIC_annToMask(obj,height,width).reshape(-1) for obj in target]
            masks = np.vstack(masks)
            masks = masks.reshape(-1, height, width)
        
        if self.target_transform is not None and len(target) > 0:
            target = self.target_transform(target, width, height)
        if self.transform is not None:
            if len(target) > 0:
                target = np.array(target)
                img, masks, boxes, labels, semantic_annotated_area = self.transform(img, masks, target[:, :4],
                    {'num_crowds': num_crowds, 'labels': target[:, 4]}, semantic_annotated_area)
            
                # I stored num_crowds in labels so I didn't have to modify the entirety of augmentations
                num_crowds = labels['num_crowds']
                labels     = labels['labels']
                
                target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
            else:
                if self.has_gt:
                    img, _, _, _, semantic_annotated_area = self.transform(img, np.zeros((1, height, width), dtype=np.float), np.array([[0, 0, 1, 1]]).astype(float),
                         {'num_crowds': 0, 'labels': np.array([0])}, np.zeros((1, height, width), dtype=np.float))
                else:
                    img, _, _, _, _ = self.transform(img, np.zeros((1, height, width), dtype=np.float), np.array([[0, 0, 1, 1]]),
                         {'num_crowds': 0, 'labels': np.array([0])}, np.zeros((1, height, width), dtype=np.float))
                    semantic_annotated_area = None
                #masks = None
                #target = None
                masks = np.zeros((1,height,width))
                target = []

        #if target.shape[0] == 0:
        #    print('Warning: Augmentation output an example with no ground truth. Resampling...')
        #    return self.pull_item(random.randint(0, len(self.ids)-1))
        
        if 'UM' in cfg.name:
            UM_class = self.UM_class_generation(path) if self.UM_classification else 0
        if 'CC' in cfg.name:
            UM_class = self.CC_class_generation(path) if self.UM_classification else 0
        
        if self.proj_slide:
            return torch.from_numpy(img).permute(2, 0, 1), target, masks, height, width, num_crowds, semantic_annotated_area, UM_class, file_name
        else:
            return torch.from_numpy(img).permute(2, 0, 1), target, masks, height, width, num_crowds, semantic_annotated_area, UM_class

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            cv2 img
        '''
        img_id = self.ids[index]
        path = self.coco.loadImgs(img_id)[0]['file_name']
        return cv2.imread(osp.join(self.root, path), cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        return self.coco.loadAnns(ann_ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_stra

    def UMSLIC_annToMask(self, obj, height, width):
        mask = np.zeros((height,width))
        inds = obj['segmentation']
        for _x, _y in zip(inds[0],inds[1]):
            mask[_x,_y] = 1
        return mask

    def semantic_annotated_area_generation(self, semantic_seg_anno_dict, height, width):
        mask = np.zeros((1,height, width))
        good_inds = semantic_seg_anno_dict['good']
        bad_inds = semantic_seg_anno_dict['bad']
        total_inds = [good_inds[0] + bad_inds[0], good_inds[1] + bad_inds[1]]
        for _x, _y in zip(total_inds[0],total_inds[1]):
            mask[0,_x,_y] = 1
        return mask

    def UM_class_generation(self,img_path):
        slide_ind = img_path.split('/')[-2].split(' ')[-1]
        return int(self.UM_class_dict[slide_ind])

    def CC_class_generation(self,img_path):
        slide_name = img_path.split('/')[-2]
        return int(self.UM_class_dict[slide_name])
def enforce_size(img, targets, masks, num_crowds, new_w, new_h,semantic_annotated_area=None):
    """ Ensures that the image is the given size without distorting aspect ratio. """
    with torch.no_grad():
        _, h, w = img.size()

        if h == new_h and w == new_w:
            return img, targets, masks, num_crowds, semantic_annotated_area
        
        # Resize the image so that it fits within new_w, new_h
        w_prime = new_w
        h_prime = h * new_w / w

        if h_prime > new_h:
            w_prime *= new_h / h_prime
            h_prime = new_h

        w_prime = int(w_prime)
        h_prime = int(h_prime)

        # Do all the resizing
        img = F.interpolate(img.unsqueeze(0), (h_prime, w_prime), mode='bilinear', align_corners=False)
        img.squeeze_(0)

        # Act like each object is a color channel
        masks = F.interpolate(masks.unsqueeze(0), (h_prime, w_prime), mode='bilinear', align_corners=False)
        masks.squeeze_(0)

        if not semantic_annotated_area is None:
            semantic_annotated_area = F.interpolate(semantic_annotated_area.unsqueeze(0), (h_prime, w_prime), mode='bilinear', align_corners=False)
            semantic_annotated_area.squeeze_(0)

        # Scale bounding boxes (this will put them in the top left corner in the case of padding)
        targets[:, [0, 2]] *= (w_prime / new_w)
        targets[:, [1, 3]] *= (h_prime / new_h)

        # Finally, pad everything to be the new_w, new_h
        pad_dims = (0, new_w - w_prime, 0, new_h - h_prime)
        img   = F.pad(  img, pad_dims, mode='constant', value=0)
        masks = F.pad(masks, pad_dims, mode='constant', value=0)
        semantic_annotated_area = F.pad(semantic_annotated_area, pad_dims, mode='constant', value=0)

        return img, targets, masks, num_crowds, semantic_annotated_area
        



def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and (lists of annotations, masks)

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list<tensor>, list<tensor>, list<int>) annotations for a given image are stacked
                on 0 dim. The output gt is a tuple of annotations and masks.
    """
    targets = []
    imgs = []
    masks = []
    num_crowds = []
    semantic_annotated_area = []
    UM_class = []

    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1][0]))
        masks.append(torch.FloatTensor(sample[1][1]))
        num_crowds.append(sample[1][2])
        semantic_annotated_area.append(sample[1][3])
        UM_class.append(sample[1][4])

    return imgs, (targets, masks, num_crowds, semantic_annotated_area, UM_class)

