#!/usr/bin/env python


#############################################################################
##
## Copyright (C) 2013 Riverbank Computing Limited.
## Copyright (C) 2010 Nokia Corporation and/or its subsidiary(-ies).
## All rights reserved.
##
## This file is part of the examples of PyQt.
##
## $QT_BEGIN_LICENSE:BSD$
## You may use this file under the terms of the BSD license as follows:
##
## "Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions are
## met:
##   * Redistributions of source code must retain the above copyright
##     notice, this list of conditions and the following disclaimer.
##   * Redistributions in binary form must reproduce the above copyright
##     notice, this list of conditions and the following disclaimer in
##     the documentation and/or other materials provided with the
##     distribution.
##   * Neither the name of Nokia Corporation and its Subsidiary(-ies) nor
##     the names of its contributors may be used to endorse or promote
##     products derived from this software without specific prior written
##     permission.
##
## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
## "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
## LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
## A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
## OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
## SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
## LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
## DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
## THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
## (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
## OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
## $QT_END_LICENSE$
##
#############################################################################


from PyQt5.QtCore import (QDir, QIODevice, QFile, QFileInfo, Qt, QTextStream,
        QUrl)
from PyQt5 import QtGui,QtCore
from PyQt5.QtGui import QDesktopServices,QImage,QPixmap,qRgb, QPainter
from PyQt5.QtWidgets import (QAbstractItemView, QApplication, QComboBox,
        QDialog, QFileDialog, QGridLayout, QHBoxLayout, QHeaderView, QLabel,
        QProgressDialog, QPushButton, QSizePolicy, QTableWidget,
        QTableWidgetItem, QWidget, QGraphicsView, QGraphicsScene)
import os
from skimage.io import imread,imsave
from skimage.transform import resize
import numpy as np 
from copy import deepcopy
import json
from time import gmtime, strftime
class ClickLabel(QLabel):
    clicked = QtCore.pyqtSignal()

    def mousePressEvent(self, event):
        self.clicked.emit()
        QLabel.mousePressEvent(self, event)


class Window(QDialog):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        browse_NetworkInputButton = self.createButton("&Browse...", self.browse_NetworkInput)
        browse_TileButton = self.createButton("&Browse...", self.browse_Tile)
        browse_OutputButton = self.createButton("&Browse...", self.browse_Output)

        StartButton = self.createButton("&Start", self.start)
        ConfirmButton = self.createButton("&Confirm", self.confirm)
        FinishButton = self.createButton("&Finish", self.finish)

        self.directoryComboBox_NetworkInput = self.createComboBox(QDir.currentPath())
        self.directoryComboBox_Tile = self.createComboBox(QDir.currentPath())
        self.directoryComboBox_Output = self.createComboBox(QDir.currentPath())

        directoryLabel_Tile = QLabel("Tile Image path:")
        directoryLabel_NetworkInput = QLabel("Network Input path:")
        directoryLabel_Output = QLabel("Output path:")
        windowExample = QWidget()
        self.IterationLabel = QLabel(windowExample)

        self.image_0 = ClickLabel()        
        self.image_1 = ClickLabel()
        self.image_2 = ClickLabel()
        self.image_3 = ClickLabel()
        self.image_4 = ClickLabel()
        self.image_5 = ClickLabel()
        self.image_6 = ClickLabel()
        self.image_7 = ClickLabel()
        self.image_8 = ClickLabel()
        self.image_9 = ClickLabel()
        for i in range(10): getattr(self,'image_'+str(i)).clicked.connect(lambda x=i: self.image_click(x))


        mainLayout = QGridLayout()
        mainLayout.addWidget(directoryLabel_NetworkInput, 0, 0)
        mainLayout.addWidget(directoryLabel_Tile, 1, 0)
        mainLayout.addWidget(directoryLabel_Output, 2, 0)

        mainLayout.addWidget(self.directoryComboBox_NetworkInput, 0, 1,1,4)
        mainLayout.addWidget(self.directoryComboBox_Tile, 1, 1,1,4)
        mainLayout.addWidget(self.directoryComboBox_Output, 2, 1,1,4)

        mainLayout.addWidget(browse_NetworkInputButton, 0, 5,1,1)
        mainLayout.addWidget(browse_TileButton, 1, 5,1,1)
        mainLayout.addWidget(browse_OutputButton, 2, 5,1,1)

        # mainLayout.addWidget(self.filesTable, 1, 0,7,1)
        mainLayout.addWidget(self.IterationLabel, 3, 0,1,1)
        mainLayout.addWidget(StartButton, 3, 1,1,1)
        mainLayout.addWidget(ConfirmButton, 3, 2,1,1)
        mainLayout.addWidget(FinishButton, 3, 3,1,1)

        mainLayout.addWidget(self.image_0, 4, 1, 1, 1)
        mainLayout.addWidget(self.image_1, 4, 2, 1, 1)
        mainLayout.addWidget(self.image_2, 4, 3, 1, 1)
        mainLayout.addWidget(self.image_3, 4, 4, 1, 1)
        mainLayout.addWidget(self.image_4, 5, 1, 1, 1)
        mainLayout.addWidget(self.image_5, 5, 2, 1, 1)
        mainLayout.addWidget(self.image_6, 5, 3, 1, 1)
        mainLayout.addWidget(self.image_7, 5, 4, 1, 1)
        mainLayout.addWidget(self.image_8, 6, 2, 1, 1)
        mainLayout.addWidget(self.image_9, 6, 3, 1, 1)

        # mainLayout.addLayout(buttonsLayout, 3, 0, 1, 4)
        self.setLayout(mainLayout)

        self.setWindowTitle("Find Files")
        self.resize(1000, 1000)

        self.max_iter = 10
        self.min_cluster_selection_num = 5
        self.stop_sample_num=20
        # Initialization
        self.para_initialization()

    def para_initialization(self):
        self.iter = 0
        self.temp_cluster = [0 for i in range(10)] # cluster_index now showing images.
        self.temp_result = [0 for i in range(10)] # the selection result for shown images.
        self.stat = {} # recorded selection for all clusters.
        for i in range(100):
            self.stat[i] = []

    def browse(self,phase=None):
        print(phase)        
        if phase != 'Tile' and phase != 'Output':
            directory = QFileDialog.getOpenFileName(self, "Find Files",
                     QDir.currentPath())
            if directory:
                if getattr(self,'directoryComboBox_'+phase).findText(directory[0]) == -1:
                    getattr(self,'directoryComboBox_'+phase).addItem(directory[0])
                getattr(self,'directoryComboBox_'+phase).setCurrentIndex(getattr(self,'directoryComboBox_'+phase).findText(directory[0]))
            
            path = getattr(self,'directoryComboBox_'+phase).currentText()
            with open(path,'r') as a:
                content = json.load(a)

            setattr(self,phase,content)
            if phase == 'NetworkInput':
                self.M = np.array(getattr(self,phase)['M'])
                self.assignment = getattr(self,phase)['assignment']
                self.assignment_info_generation()
                self.human_initial_point = getattr(self,phase)['human_initial_point']


        else:
            directory = QFileDialog.getExistingDirectory(self, "Find Files",
                    QDir.currentPath())
            if directory:
                if getattr(self,'directoryComboBox_'+phase).findText(directory) == -1:
                    getattr(self,'directoryComboBox_'+phase).addItem(directory)
                getattr(self,'directoryComboBox_'+phase).setCurrentIndex(getattr(self,'directoryComboBox_'+phase).findText(directory))
            
            setattr(self,phase+'_dir',getattr(self,'directoryComboBox_'+phase).currentText())
            print(phase+'_dir',getattr(self,phase+'_dir'))
        
    def browse_NetworkInput(self):
        self.browse('NetworkInput')
        self.para_initialization()

    def browse_Tile(self):
        self.browse('Tile')
        self.para_initialization()

    def browse_Output(self):
        self.browse('Output')

    def assignment_info_generation(self):
        self.img_name = list(self.assignment.keys())
        self.assignment_array = []

        for name in self.img_name:
            self.assignment_array.append(self.assignment[name])
        self.assignment_array = np.array(self.assignment_array)
        #self.img_name[0] corresponds to self.assignment_array[0]

        self.assignment_dict = {}
        for i in range(100):
            a1,a2,a3 = np.where(self.assignment_array == i)
            a1 = a1.tolist()
            a2 = a2.tolist()
            a3 = a3.tolist()
            self.assignment_dict[i] = list(zip(a1,a2,a3)) # get the index, [0] means the img, [1,2] is the location
        self.M_valid = [i for i in self.assignment_dict if len(self.assignment_dict[i]) > 0]
        self.potential_M = self.M_valid*self.stop_sample_num
        self.used_index = []

    def start(self):
        #Initialization
        self.para_initialization()
        self.feature_point = self.human_initial_point
        # Show images
        self.temp_cluster = self.select_clusters(all_False=False)
        # show images
        self.show_images(self.temp_cluster)

        self.IterationLabel.setText('Iter 1')


    def random_cluster_selection(self,selected_clusters):
        remaining_index = [i for i in range(len(self.potential_M)) if i not in self.used_index]
        remaining_clusters = [i for i in self.M_valid if len(self.stat[i]) < self.stop_sample_num]
        print(len(selected_clusters),len(remaining_clusters))
        random_index = np.random.choice(remaining_index,size=10, replace=False)
        random_clusters = [self.potential_M[i] for i in random_index]
        self.used_index += random_index.tolist()
        return random_clusters

    def select_clusters(self,all_False):
        # if all_False:
        clusters_to_use = self.random_cluster_selection([])
        return np.array(clusters_to_use).astype(np.uint8)

    def show_images(self,cluster_to_use):
        # First select patch image
        for i,cluster_num in enumerate(cluster_to_use):
            # get info
            randint = np.random.randint(len(self.assignment_dict[cluster_num]))
            selected_info = self.assignment_dict[cluster_num][randint]
            image_name = self.img_name[selected_info[0]]
            index_x = selected_info[1]
            index_y = selected_info[2]
            # load image
            tile_image = imread(os.path.join(self.Tile_dir,image_name))
            tile_height, tile_width, self.channel = tile_image.shape
            #prepare patch image to show
            self.patch_height = tile_height//4*2
            self.patch_width = tile_width//4*2
            bytesPerLine = 3 * self.patch_width
            setattr(self,'patch_'+str(i),deepcopy(tile_image[index_x*(self.patch_height//2):(index_x+2)*(self.patch_height//2),index_y*(self.patch_width//2):(index_y+2)*(self.patch_width//2),:]))
            qImg = QImage(getattr(self,'patch_'+str(i)), self.patch_width,self.patch_height,bytesPerLine,QImage.Format_RGB888)
            getattr(self,'image_'+str(i)).setPixmap(QPixmap(qImg).scaled(128,128))
    def show_black_images(self):
        for i in range(10):
            qImg = QImage(np.zeros((self.patch_height,self.patch_width,self.channel)), self.patch_width,self.patch_height,3*self.patch_width,QImage.Format_RGB888)
            getattr(self,'image_'+str(i)).setPixmap(QPixmap(qImg).scaled(128,128))

    def confirm(self):
        # First record to stat
        self.stat_update()
        # second update feature point
        self.feature_point_update()
        # Then intialization for next iteration
        all_False = np.sum(self.temp_result) == 0
        print(all_False)
        self.iter += 1
        self.IterationLabel.setText('Iter '+str(self.iter+1)+'/200')
        self.temp_result = [0 for i in range(10)]
        
        # Finally show images for next iteration
        # if all clusters has more than 10 samples, finish
        self.finish_bool = self.finish_criterion()
        if not self.finish_bool:
            self.temp_cluster = self.select_clusters(all_False)
            self.show_images(self.temp_cluster)
        else:
            self.show_black_images()
            self.temp_cluster = []

    def stat_update(self):
        for i in range(len(self.temp_cluster)):
            cluster_ind = self.temp_cluster[i]
            print(cluster_ind)
            self.stat[cluster_ind].append(self.temp_result[i])

    def feature_point_update(self,feature_point_update_multi = 0.5):
        if np.sum(self.temp_result) > 0:
            used_centroid_vector = self.M[:,self.temp_cluster[np.where(np.array(self.temp_result) == 1)[0]]]
            print(used_centroid_vector.shape)# Below calculation BIAS
            for i in range(used_centroid_vector.shape[1]):
                self.feature_point = self.feature_point + (used_centroid_vector[:,i]-self.feature_point)*feature_point_update_multi/10

    def finish_criterion(self):
        remaining_clusters = [i for i in self.M_valid if len(self.stat[i]) < self.stop_sample_num]
        bool_ = len(remaining_clusters) == 0
        return bool_

    def finish(self):
        # check if max iter
        if self.finish_bool:
            day_time = strftime("%d_%b_%Y_%H_%M_%S", gmtime())
            with open(os.path.join(self.Output_dir,day_time+'.json'),'w') as a:
                json.dump(self.stat,a,indent=4)



    @staticmethod
    def updateComboBox(comboBox):
        if comboBox.findText(comboBox.currentText()) == -1:
            comboBox.addItem(comboBox.currentText())

    def createButton(self, text, member):
        button = QPushButton(text)
        button.clicked.connect(member)
        return button

    def createComboBox(self, text=""):
        comboBox = QComboBox()
        comboBox.setEditable(True)
        comboBox.addItem(text)
        comboBox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        return comboBox

    def margin_generation(self,tile_image,width=10):
        tile_image[:width,:,0] = 0
        tile_image[:width,:,1] = 0
        tile_image[:width,:,2] = 255
        
        tile_image[-width:,:,0] = 0
        tile_image[-width:,:,1] = 0
        tile_image[-width:,:,2] = 255
        
        tile_image[:,:width,0] = 0
        tile_image[:,:width,1] = 0
        tile_image[:,:width,2] = 255
        
        tile_image[:,-width:,0] = 0
        tile_image[:,-width:,1] = 0
        tile_image[:,-width:,2] = 255

        return tile_image

    def image_click(self,i):
        patch_image = deepcopy(getattr(self,'patch_'+str(i)))
        patch_height, patch_width, channel = patch_image.shape
        bytesPerLine = 3 * patch_width

        if self.temp_result[i] == 0:
            patch_image = deepcopy(self.margin_generation(patch_image))
        qImg = QImage(patch_image, patch_width,patch_height, bytesPerLine,QImage.Format_RGB888)
        getattr(self,'image_'+str(i)).setPixmap(QPixmap(qImg).scaled(128,128))

        self.temp_result[i] = 1 - self.temp_result[i]

        print(self.temp_result)








if __name__ == '__main__':

    import sys

    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())