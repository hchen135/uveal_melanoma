import torch
from glob import glob
import os
import json
import numpy as np
class CellDataset(torch.utils.data.Dataset):
    def __init__(self, subject_dict, DATA_DIR,batch_size,max_cells=1000):
        self.subject_dict = subject_dict
        self.max_cells = max_cells
        self.batch_size = batch_size
        self.cell_data = []
        self.labels = []
        self.names = []
        for subject in subject_dict:
            try:
                with open(os.path.join(DATA_DIR,"feature_"+str(subject.split(' ')[1])+"_info.json")) as a:
                    content = json.load(a)

                print(subject, "loaded")

                features = content['features']
                self.cell_data.append(features)
                self.labels.append(subject_dict[subject])
                self.names.append(subject)
            except:
                pass

        print(len(self.cell_data))
        self.out_global = 0

    def __len__(self):
        return len(self.cell_data)

    def get_cells(self,cell_list,phase):
        # print(len(cell_list))
        if len(cell_list) <= self.max_cells or phase == "test":
            return cell_list
        else:
            index = np.random.choice(len(cell_list),self.max_cells,replace=True)
            len(index)
            return [cell_list[i] for i in index]

    def out(self,phase="train",index = None):
        if index is None:
            if phase == "train":
                index = np.random.randint(len(self.cell_data),size=self.batch_size)
            else:
                index = [self.out_global]
                self.out_global += 1
        else:
            index = [index]
        cells = [self.get_cells(self.cell_data[i],phase) for i in index]
        return cells, [self.labels[i] for i in index], [self.names[i] for i in index]
                
    def __getitem__(self, index):
        print("index:",index)
        return self.get_cells(self.cell_data[index]), self.labels[index]