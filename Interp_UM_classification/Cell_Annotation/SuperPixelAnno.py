from PyQt5 import QtCore, QtGui, QtWidgets

from PyQt5.QtCore import (QDir, QIODevice, QFile, QFileInfo, Qt, QTextStream,
        QUrl)
from PyQt5 import QtGui,QtCore
from PyQt5.QtGui import QDesktopServices,QImage,QPixmap,qRgb, QPainter
from PyQt5.QtWidgets import (QAbstractItemView, QApplication, QComboBox,
        QDialog, QFileDialog, QGridLayout, QHBoxLayout, QHeaderView, QLabel,
        QProgressDialog, QPushButton, QSizePolicy, QTableWidget,
        QTableWidgetItem, QWidget, QGraphicsView, QGraphicsScene,QGraphicsItem)

from openslide import OpenSlide
from PIL.ImageQt import ImageQt
from PIL import Image
import json
import numpy as np
from util import *
from copy import deepcopy
# from time import gmtime, strftime
import os
import time
from skimage.io import imread


class PhotoViewer(QtWidgets.QGraphicsView):
    photoClicked = QtCore.pyqtSignal(QtCore.QPoint)

    def __init__(self, parent):
        super(PhotoViewer, self).__init__(parent)
        self._zoom = 0
        self._empty = True
        self._scene = QtWidgets.QGraphicsScene(self)
        self._photo = QtWidgets.QGraphicsPixmapItem()
        self._scene.addItem(self._photo)
        self.setScene(self._scene)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(30, 30, 30)))
        self.setFrameShape(QtWidgets.QFrame.NoFrame)

        # self.horizontalScrollBar().disconnect()
        # self.verticalScrollBar().disconnect()

        self._isPanning = False
        self._mousePressedRight = False

        # self._rect = QtCore.QRectF(0, 0, 10000, 10000)
    def hasPhoto(self):
        return not self._empty

    def fitInView(self, scale=True):
        rect_image = QtCore.QRectF(self._photo.pixmap().rect())
        # print(('rect_image', rect_image))
        rect = QtCore.QRectF(self.rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            if self.hasPhoto():
                unity = self.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
                self.scale(1 / unity.width(), 1 / unity.height())
                viewrect = self.viewport().rect()
                scenerect = self.transform().mapRect(rect_image)
                factor = min(viewrect.width() / scenerect.width(),
                             viewrect.height() / scenerect.height())
                self.scale(factor, factor)
                # print('viewrect',viewrect)
                # print('scenerect',scenerect)
            self._zoom = 0
    def setPhoto(self, pixmap=None):
        if pixmap:
            self.pixmap=pixmap
        self._zoom = 0
        if pixmap and not pixmap.isNull():
            self._empty = False
            # self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
            self._photo.setPixmap(pixmap)
        else:
            self._empty = True
            # self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
            self._photo.setPixmap(QtGui.QPixmap())
        self.fitInView()




class Window(QtWidgets.QWidget):
    def __init__(self):
        super(Window, self).__init__()

        self.viewer1 = PhotoViewer(self)
        self.viewer2 = PhotoViewer(self)

        self.viewer1.setAcceptDrops(True)
        self.viewer1.setMouseTracking(True)
        # self.setMouseTracking(True)
        self.viewer2.setAcceptDrops(True)
        self.viewer2.setMouseTracking(True)
        self.viewer1.viewport().installEventFilter(self)
        self.viewer2.viewport().installEventFilter(self)


        browse_ImagePathButton = self.createButton("&Browse...", self.browse_ImagePath)
        LoadImageButton = self.createButton("&Load Image", self.loadImage)
        FinishButton = self.createButton("&Finish and save...", self.finish)
        GoodButton = self.createButton("&Good", self.Good)
        BadButton = self.createButton("&Bad", self.Bad)

        self.directoryComboBox_ImagePath = self.createComboBox(QDir.currentPath())

        directoryLabel_ImagePath = QLabel("Image Path:")


        # Arrange layout
        VBlayout = QtWidgets.QVBoxLayout(self)
        HBlayout0 = QtWidgets.QHBoxLayout()
        HBlayout0.setAlignment(QtCore.Qt.AlignLeft)
        HBlayout0.addWidget(self.viewer1)
        HBlayout0.addWidget(self.viewer2)
        VBlayout.addLayout(HBlayout0)

        VBlayout1 = QtWidgets.QVBoxLayout()

        HBlayout2 = QtWidgets.QHBoxLayout()
        HBlayout2.setAlignment(QtCore.Qt.AlignLeft)
        HBlayout2.addWidget(directoryLabel_ImagePath)
        HBlayout2.addWidget(self.directoryComboBox_ImagePath)
        HBlayout2.addWidget(browse_ImagePathButton)
        VBlayout1.addLayout(HBlayout2)

        HBlayout6 = QtWidgets.QHBoxLayout()
        HBlayout6.setAlignment(QtCore.Qt.AlignRight)
        HBlayout6.addWidget(GoodButton)
        HBlayout6.addWidget(BadButton)
        HBlayout6.addWidget(LoadImageButton)
        HBlayout6.addWidget(FinishButton)
        VBlayout1.addLayout(HBlayout6)

        VBlayout.addLayout(VBlayout1)

        self.loaded_img_bool=False
        self.image_path = None

        self.anno_all = {}
        self.anno_all['good'] = []
        self.anno_all['bad'] = []
        self.anno_temp = []

    def loadImage(self):

        # self.viewer.setPhoto(QtGui.QPixmap('/Users/hc/Documents/Uveal Melanoma/Slide 10/1233.png'))
        self.InitializationPreperation()

        qim = ImageQt(deepcopy(Image.fromarray(self.image)))
        qim_state = ImageQt(deepcopy(Image.fromarray(self.SLIC_visualization)))
        self.viewer1.setPhoto(QPixmap(QtGui.QPixmap.fromImage(qim)))
        self.viewer2.setPhoto(QPixmap(QtGui.QPixmap.fromImage(qim_state)))
        
        self.loaded_img_bool = True
        self.anno_all = {}
        self.anno_all['good'] = []
        self.anno_all['bad'] = []
        self.anno_temp = []

    def done_color(self,labels,r,g,b):
        for i in labels:
            self.SLIC_visualization[:,:,0][self.SLIC == i] = r
            self.SLIC_visualization[:,:,1][self.SLIC == i] = g
            self.SLIC_visualization[:,:,2][self.SLIC == i] = b 
        print('color changed')
        qim_state = ImageQt(deepcopy(Image.fromarray(self.SLIC_visualization)))
        self.viewer2.setPhoto(QPixmap(QtGui.QPixmap.fromImage(qim_state)))
        self.viewer2.horizontalScrollBar().setValue(0)
        self.viewer2.verticalScrollBar().setValue(0)

    def Good(self):
        self.anno_all['good'].append(list(set(self.anno_temp)))

        self.done_color(self.anno_temp,255,255,0)

        self.anno_temp = []
        print(self.anno_all)

    def Bad(self):
        self.anno_all['bad'].append(list(set(self.anno_temp)))

        self.done_color(self.anno_temp,0,0,255)

        self.anno_temp = []
        print(self.anno_all)

    def pixInfo(self):
        self.viewer1.toggleDragMode()
        self.viewer2.toggleDragMode()

    def browse(self,phase=None):
        print(phase)        
        if phase in ['ImagePath']:
            # if self.image_path:
            #     directory = QFileDialog.getOpenFileName(self, "Find Files",
            #              self.image_path)
            # else:
            #     directory = QFileDialog.getOpenFileName(self, "Find Files",
            #              QDir.currentPath())
            directory = QFileDialog.getOpenFileName(self, "Find Files",
                    '/Users/hc/Documents/JHU/PJ/Mathias/microscopy/Nature/Cervical_cancer/data/generated_data/SLIC_anno_200_0.1')
            if directory:
                if getattr(self,'directoryComboBox_'+phase).findText(directory[0]) == -1:
                    getattr(self,'directoryComboBox_'+phase).addItem(directory[0])
                getattr(self,'directoryComboBox_'+phase).setCurrentIndex(getattr(self,'directoryComboBox_'+phase).findText(directory[0]))
            
            path = getattr(self,'directoryComboBox_'+phase).currentText()
            print(path)
            self.image_path = '/'.join(path.split('/')[:-1])
            self.image_name = path.split('/')[-1].split('.')[0]
            
            if phase == 'ImagePath':
                self.image = imread(path)
                self.SLIC = np.load('/'.join(path.split('/')[:-1])+'/'+path.split('/')[-1].split('.')[0]+'.npy')
        self.loadImage()

        # else:
        #     directory = QFileDialog.getExistingDirectory(self, "Find Files",
        #             QDir.currentPath())
        #     if directory:
        #         if getattr(self,'directoryComboBox_'+phase).findText(directory) == -1:
        #             getattr(self,'directoryComboBox_'+phase).addItem(directory)
        #         getattr(self,'directoryComboBox_'+phase).setCurrentIndex(getattr(self,'directoryComboBox_'+phase).findText(directory))
            
        #     setattr(self,phase+'_dir',getattr(self,'directoryComboBox_'+phase).currentText())
        #     print(phase+'_dir',getattr(self,phase+'_dir'))

    def browse_ImagePath(self):
        self.browse('ImagePath')

    def InitializationPreperation(self):
        # create the SLIC image
        self.SLIC_visualization = np.zeros_like(self.image)
        for i in np.unique(self.SLIC):
            for j in range(3):
                self.SLIC_visualization[:,:,j][self.SLIC == i] = np.average(self.image[:,:,j][self.SLIC == i])
        self.SLIC_visualization_ori = deepcopy(self.SLIC_visualization)

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

    def eventFilter(self, object, event):
        # print(event.type())
        if event.type() == QtCore.QEvent.MouseButtonPress:
            if event.button() == Qt.RightButton:                
                if event.globalX() >= self.viewer2.x():  
                    scale_factor = self.viewer2.transform().m11()
                    canvas_x = self.viewer2.horizontalScrollBar().value() + event.x()
                    canvas_y = self.viewer2.verticalScrollBar().value() + event.y()

                    print(canvas_x,canvas_y,canvas_x//scale_factor,canvas_y//scale_factor)
                    label = self.SLIC[int(canvas_y//scale_factor),int(canvas_x//scale_factor)]
                    if label not in self.anno_temp:
                        self.done_color([label],0,255,0)
                        self.anno_temp.append(int(label))
                    else:
                        _x = int(canvas_y//scale_factor)
                        _y = int(canvas_x//scale_factor)
                        _r = self.SLIC_visualization_ori[_x,_y,0]
                        _g = self.SLIC_visualization_ori[_x,_y,1]
                        _b = self.SLIC_visualization_ori[_x,_y,2]
                        self.done_color([label],_r,_g,_b)
                        while label in self.anno_temp:
                            self.anno_temp.remove(label)
                    print(self.anno_temp)
                    self.viewer2.horizontalScrollBar().setValue(0)
                    self.viewer2.verticalScrollBar().setValue(0)
        return False
    
    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Q:
            self.Good()
        elif e.key() == Qt.Key_E:
            self.Bad()    

    def buildExamplePopup(self,img_extract):
        # name = item.text()
        exPopup = examplePopup(img_extract)
        self.exPopup = exPopup
        self.exPopup.got_annotation.connect(self.got_single_annotation)
        # self.exPopup.setGeometry(QtCore.QRect(100, 100, 600, 600))
        self.exPopup.show()


    def finish(self):
        with open(os.path.join(self.image_path,self.image_name+'.anno'),'w') as a:
                json.dump(self.anno_all,a,indent=4)
        self.anno_all = {}
        self.anno_all['good'] = []
        self.anno_all['bad'] = []
        self.anno_temp = []
        self.SLIC_visualization = None
        self.image = None



if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.setGeometry(200, 100, 1600, 1000)
    window.show()
    sys.exit(app.exec_())