from PyQt5 import QtCore, QtGui, QtWidgets

from PyQt5.QtCore import (QDir, QIODevice, QFile, QFileInfo, Qt, QTextStream,
        QUrl)
from PyQt5 import QtGui,QtCore
from PyQt5.QtGui import QDesktopServices,QImage,QPixmap,qRgb, QPainter, QFont
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

        self._rect = QtCore.QRectF(0, 0, 10000, 10000)
    def hasPhoto(self):
        return not self._empty

    def fitInView(self, scale=True):
        rect_image = QtCore.QRectF(self._photo.pixmap().rect())
        # print(('rect_image', rect_image))
        # rect = QtCore.QRectF(self.rect())
        if not self._rect.isNull():
            self.setSceneRect(self._rect)
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

    def wheelEvent(self, event):
        print("Enter Photo Wheel Event!", self.hasPhoto())
        if self.hasPhoto():
            
            if event.angleDelta().y() > 0:
                factor = 1.25
                self._zoom += 1
            else:
                factor = 0.8
                self._zoom -= 1
            print('y',event.angleDelta().y(),self._zoom)
            if self._zoom > 0:
                self.scale(factor, factor)
                print("scale", factor, time.time())
            elif self._zoom == 0:
                # self.fitInView()
                self.fitInView()
                self.centerOn(QtCore.QPointF(0, 0))
            else:
                self._zoom = 0

    def toggleDragMode(self):
        if self.dragMode() == QtWidgets.QGraphicsView.ScrollHandDrag:
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        elif not self._photo.pixmap().isNull():
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)

    def mousePressEvent(self,  event):            
        if event.button() == Qt.RightButton:
            self._mousePressedRight = True
            self._dragPos = event.pos()
            event.accept()
        else:
            super(PhotoViewer, self).mousePressEvent(event)
    def mouseMoveEvent(self, event):
        if self._mousePressedRight:
            newPos = event.pos()
            diff = newPos - self._dragPos
            self._dragPos = newPos
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - diff.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - diff.y())
            event.accept()
        else:
            # print('mouse move in photo')
            super(PhotoViewer, self).mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton:
            self._mousePressedRight = False
        super(PhotoViewer, self).mouseReleaseEvent(event)




class Window(QtWidgets.QWidget):
    def __init__(self):
        super(Window, self).__init__()
        self.lambda_2 = 0.5
        self.lambda_1 = 0.2

        self.viewer1 = PhotoViewer(self)
        self.viewer2 = PhotoViewer(self)
        self.viewer3 = PhotoViewer(self)
        # # 'Load image' button
        # self.btnLoad = QtWidgets.QToolButton(self)
        # self.btnLoad.setText('Load image')
        # self.btnLoad.clicked.connect(self.loadImage)
        # # Button to change from drag/pan to getting pixel info
        # self.btnPixInfo = QtWidgets.QToolButton(self)
        # self.btnPixInfo.setText('Enter pixel info mode')
        # self.btnPixInfo.clicked.connect(self.pixInfo)
        # self.editPixInfo = QtWidgets.QLineEdit(self)
        # self.editPixInfo.setReadOnly(True)
        self.viewer1.photoClicked.connect(self.photoClicked)
        self.viewer2.photoClicked.connect(self.photoClicked)

        self.viewer1.setAcceptDrops(True)
        self.viewer1.setMouseTracking(True)
        # self.setMouseTracking(True)
        self.viewer2.setAcceptDrops(True)
        self.viewer2.setMouseTracking(True)
        self.viewer1.viewport().installEventFilter(self)
        self.viewer2.viewport().installEventFilter(self)


        browse_NetworkInputButton = self.createButton("&Browse...", self.browse_NetworkInput)
        browse_ClusterSelectionButton = self.createButton("&Browse...", self.browse_ClusterSelection)
        browse_SlideButton = self.createButton("&Browse...", self.browse_Slide)
        browse_OutputButton = self.createButton("&Browse...", self.browse_Output)
        LoadImageButton = self.createButton("&Load Image", self.loadImage)
        InitializationPreperationButton = self.createButton("&Initialization Preparation...", self.InitializationPreperation)
        FinishButton = self.createButton("&Finish and save...", self.finish)

        windowExample = QWidget()
        self.IterationLabel = QLabel(windowExample)
        self.IterationLabel.setText('Number of high-quality regions now: 0')

        self.directoryComboBox_NetworkInput = self.createComboBox(QDir.currentPath())
        self.directoryComboBox_ClusterSelection = self.createComboBox(QDir.currentPath())
        self.directoryComboBox_Slide = self.createComboBox(QDir.currentPath())
        self.directoryComboBox_Output = self.createComboBox(QDir.currentPath())

        directoryLabel_NetworkInput = QLabel("Network Input path:")
        directoryLabel_ClusterSelection = QLabel("Cluster Selection path:")
        directoryLabel_Slide = QLabel("Slide Image path:")
        directoryLabel_Output = QLabel("Output path:")

        description_img = QLabel("Whole Slide Image"+" "*80)
        description_img.setFont(QFont('Arial', 20)) 
        description_state_img = QLabel(" "*50+"State Image")
        description_state_img.setFont(QFont('Arial', 20)) 
        description_ROI = QLabel(" "*45+"The region mouse hovering")
        description_ROI.setFont(QFont('Arial', 20)) 

        initialization_instructions = QLabel("How to initialize: \n"+
            "1. Load network feature results for the corresponding slide image.\n"+
            "2. Load centroid selection results.\n"+
            "3. Load corresponding slide image.\n"+
            "4. Define the output directory.\n"+
            "5. Click \"Initialization Preparation\".\n"+
            "6. Click \"Load Image\".")

        process_instructions = QLabel("How to use: \n"+
            "In either WSI or state image area: \n"+
            "1. Hover the mouse to show corresponding region in full resolution.\n"+
            "2. Right press the mouse to translate the image.\n"+
            "3. Scroll wheels to zoom-in or -out the image.\n"+
            "4. Double left click to select regions to re-annotate.\n"+
            "After re-annotation, click\"Finish and save ...\"."
            )

        self.white_example_label = QLabel(windowExample)
        self.light_gray_example_label = QLabel(windowExample)
        self.gray_example_label = QLabel(windowExample)
        self.pink_example_label = QLabel(windowExample)

        _example_size = 20
        bytesPerLine = 3 * _example_size

        white_example = (np.ones((_example_size,_example_size,3))*255).astype(np.uint8)
        white_example = QImage(white_example, _example_size, _example_size, bytesPerLine, QImage.Format_RGB888)
        # white_example = Image.fromarray(white_example.astype(np.uint8))
        # white_example.save("/Users/hc/Documents/JHU/PJ/Mathias/microscopy/Annotation_GUI/white.png")
        light_gray_example = (np.ones((_example_size,_example_size,3))*100).astype(np.uint8)
        light_gray_example = QImage(light_gray_example, _example_size, _example_size, bytesPerLine, QImage.Format_RGB888)
        gray_example = (np.ones((_example_size,_example_size,3))*50).astype(np.uint8)
        gray_example = QImage(gray_example, _example_size, _example_size, bytesPerLine, QImage.Format_RGB888)
        pink_example = (np.ones((_example_size,_example_size,3))*100).astype(np.uint8)
        pink_example[:,:,0] = 255
        pink_example = QImage(pink_example, _example_size, _example_size, bytesPerLine, QImage.Format_RGB888)

        # # _white = ImageQt(deepcopy(white_example))
        # _light_gray = ImageQt(deepcopy(light_gray_example))
        # _gray = ImageQt(deepcopy(gray_example))
        # _pink = ImageQt(deepcopy(pink_example))

        self.white_example_label.setPixmap(QPixmap(QtGui.QPixmap.fromImage(white_example)))
        self.light_gray_example_label.setPixmap(QPixmap(QtGui.QPixmap.fromImage(light_gray_example)))
        self.gray_example_label.setPixmap(QPixmap(QtGui.QPixmap.fromImage(gray_example)))
        self.pink_example_label.setPixmap(QPixmap(QtGui.QPixmap.fromImage(pink_example)))

        self.region_words = QLabel(windowExample)
        self.region_words.setText(" "*20+"Different regions: ")
        self.white_instruction = QLabel(windowExample)
        self.white_instruction.setText('High-quality'+' '*5)
        self.light_gray_instruction = QLabel(windowExample)
        self.light_gray_instruction.setText('Mix-quality'+' '*5)
        self.gray_instruction = QLabel(windowExample)
        self.gray_instruction.setText('Low-quality'+' '*5)
        self.pink_instruction = QLabel(windowExample)
        self.pink_instruction.setText('Suspicious'+' '*5)

        # Arrange layout
        VBlayout = QtWidgets.QVBoxLayout(self)
        HBlayout00 = QtWidgets.QHBoxLayout()
        HBlayout00.setAlignment(QtCore.Qt.AlignJustify)
        HBlayout00.addWidget(description_img)
        HBlayout00.addWidget(description_state_img)
        VBlayout.addLayout(HBlayout00)

        HBlayout0 = QtWidgets.QHBoxLayout()
        HBlayout0.setAlignment(QtCore.Qt.AlignLeft)
        HBlayout0.addWidget(self.viewer1)
        HBlayout0.addWidget(self.viewer2)
        VBlayout.addLayout(HBlayout0)

        HBlayout_bottom = QtWidgets.QHBoxLayout()
        HBlayout_bottom.setAlignment(QtCore.Qt.AlignLeft)
        
        VBlayout0 = QtWidgets.QVBoxLayout()
        VBlayout0.addWidget(description_ROI)
        VBlayout0.addWidget(self.viewer3)
        HBlayout_bottom.addLayout(VBlayout0)

        VBlayout1 = QtWidgets.QVBoxLayout()
        # HBlayout1 = QtWidgets.QHBoxLayout()
        # HBlayout1.setAlignment(QtCore.Qt.AlignLeft)
        # HBlayout1.addWidget(self.btnLoad)
        # HBlayout1.addWidget(self.btnPixInfo)
        # HBlayout1.addWidget(self.editPixInfo)
        # VBlayout1.addLayout(HBlayout1)

        HBlayout_color = QtWidgets.QHBoxLayout()
        HBlayout_color.setAlignment(QtCore.Qt.AlignLeft)
        HBlayout_color.addWidget(self.region_words)
        HBlayout_color.addWidget(self.white_example_label)
        HBlayout_color.addWidget(self.white_instruction)
        HBlayout_color.addWidget(self.light_gray_example_label)
        HBlayout_color.addWidget(self.light_gray_instruction)
        HBlayout_color.addWidget(self.gray_example_label)
        HBlayout_color.addWidget(self.gray_instruction)
        HBlayout_color.addWidget(self.pink_example_label)
        HBlayout_color.addWidget(self.pink_instruction)
        VBlayout1.addLayout(HBlayout_color)



        HBlayout2 = QtWidgets.QHBoxLayout()
        HBlayout2.setAlignment(QtCore.Qt.AlignLeft)
        HBlayout2.addWidget(directoryLabel_NetworkInput)
        HBlayout2.addWidget(self.directoryComboBox_NetworkInput)
        HBlayout2.addWidget(browse_NetworkInputButton)
        VBlayout1.addLayout(HBlayout2)

        HBlayout3 = QtWidgets.QHBoxLayout()
        HBlayout3.setAlignment(QtCore.Qt.AlignLeft)
        HBlayout3.addWidget(directoryLabel_ClusterSelection)
        HBlayout3.addWidget(self.directoryComboBox_ClusterSelection)
        HBlayout3.addWidget(browse_ClusterSelectionButton)
        VBlayout1.addLayout(HBlayout3)

        HBlayout4 = QtWidgets.QHBoxLayout()
        HBlayout4.setAlignment(QtCore.Qt.AlignLeft)
        HBlayout4.addWidget(directoryLabel_Slide)
        HBlayout4.addWidget(self.directoryComboBox_Slide)
        HBlayout4.addWidget(browse_SlideButton)
        VBlayout1.addLayout(HBlayout4)

        HBlayout5 = QtWidgets.QHBoxLayout()
        HBlayout5.setAlignment(QtCore.Qt.AlignLeft)
        HBlayout5.addWidget(directoryLabel_Output)
        HBlayout5.addWidget(self.directoryComboBox_Output)
        HBlayout5.addWidget(browse_OutputButton)
        VBlayout1.addLayout(HBlayout5)

        HBlayout6 = QtWidgets.QHBoxLayout()
        HBlayout6.setAlignment(QtCore.Qt.AlignRight)
        HBlayout6.addWidget(self.IterationLabel)
        HBlayout6.addWidget(InitializationPreperationButton)
        HBlayout6.addWidget(LoadImageButton)
        HBlayout6.addWidget(FinishButton)
        VBlayout1.addLayout(HBlayout6)

        HBlayout7 = QtWidgets.QHBoxLayout()
        HBlayout7.addWidget(initialization_instructions)
        HBlayout7.addWidget(process_instructions)
        VBlayout1.addLayout(HBlayout7)

        HBlayout_bottom.addLayout(VBlayout1)
        VBlayout.addLayout(HBlayout_bottom)

        self.SlideRescaleDefault=512
        self.loaded_img_bool=False



    def loadImage(self):

        # self.viewer.setPhoto(QtGui.QPixmap('/Users/hc/Documents/Uveal Melanoma/Slide 10/1233.png'))
        
        qim = ImageQt(deepcopy(self.resized_SlideImg_default))
        qim_state = ImageQt(deepcopy(self.state_img_default))
        self.viewer1.setPhoto(QPixmap(QtGui.QPixmap.fromImage(qim)))
        self.viewer2.setPhoto(QPixmap(QtGui.QPixmap.fromImage(qim_state)))
        
        self.loaded_img_bool = True
        self.IterationLabel.setText('Number of high-quality regions now: '+str(int(np.sum(np.array(self.state_img_default)[:,:,0] == 255))))


    def pixInfo(self):
        self.viewer1.toggleDragMode()
        self.viewer2.toggleDragMode()

    def photoClicked(self, pos):
        if self.viewer1.dragMode()  == QtWidgets.QGraphicsView.NoDrag:
            self.editPixInfo.setText('%d, %d' % (pos.x(), pos.y()))

    def browse(self,phase=None):
        print(phase)        
        if phase in ['NetworkInput','ClusterSelection','Slide']:
            directory = QFileDialog.getOpenFileName(self, "Find Files",
                     QDir.currentPath())
            if directory:
                if getattr(self,'directoryComboBox_'+phase).findText(directory[0]) == -1:
                    getattr(self,'directoryComboBox_'+phase).addItem(directory[0])
                getattr(self,'directoryComboBox_'+phase).setCurrentIndex(getattr(self,'directoryComboBox_'+phase).findText(directory[0]))
            
            path = getattr(self,'directoryComboBox_'+phase).currentText()
            

            
            if phase == 'NetworkInput':
                with open(path,'r') as a:
                    content = json.load(a)
                self.NetworkInput = content
                self.NetworkInputPath = path
            elif phase == 'ClusterSelection':
                with open(path,'r') as a:
                    content = json.load(a)
                self.ClusterSelection = content
            elif phase == 'Slide':
                # first open the whole slide image
                self.SlideName = path.split('/')[-1].split('.')[0]
                self.SlideImg = OpenSlide(path)
                self.modification_list = []
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

    def browse_ClusterSelection(self):
        self.browse('ClusterSelection')

    def browse_Slide(self):
        self.browse('Slide')

    def browse_Output(self):
        self.browse('Output')

    def SlideInfoExtraction(self):
        original_dimensions = self.SlideImg.dimensions
        resize_width = original_dimensions[0]//self.SlideRescaleDefault
        resize_height = original_dimensions[1]//self.SlideRescaleDefault
        # resized_img = img.get_thumbnail((resize_width,resize_height))
        return resize_height,resize_width

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


    def InitializationPreperation(self):
        # first, network input processing
        self.M = np.array(self.NetworkInput['M'])
        self.assignment = self.NetworkInput['assignment']
        # self.assignment_info_generation()
        self.human_initial_point = self.NetworkInput['human_initial_point']
        self.output_dict = self.NetworkInput['feature_vector']

        #Then Slide preparation
        # get the info of the size
        self.SlideRescaleHeight,self.SlideRescaleWidth = self.SlideInfoExtraction()
        # thumbnail of the WSI, to show in default
        self.resized_SlideImg_default = self.SlideImg.get_thumbnail((self.SlideRescaleWidth,self.SlideRescaleHeight))
        _width,_height = self.resized_SlideImg_default.size
        self.resized_SlideImg_default = self.resized_SlideImg_default.resize((_width*3,_height*3),Image.NEAREST)
        # generate an image, which i shown, together with whole slide image.
        self.state_img_default = self.state_img_generation()

        if os.path.exists(os.path.join(self.Output_dir,self.SlideName+'.json')):
            with open(os.path.join(self.Output_dir,self.SlideName+'.json')) as a:
                content = json.load(a)
            self.modification_list = content['modification']
            self.state_initial_previous_mosification()
        self.state_img_default = Image.fromarray(self.state_img_default.astype(np.uint8))

    def state_initial_previous_mosification(self):
        _color = {True:255,False:50}
        _count = 0
        for i in self.output_slide:
            tile_height,tile_width = i.split('TileLoc_')[-1].split('.')[0].split('_')[:2]
            tile_height = int(tile_height)
            tile_width = int(tile_width)
            for patch_height in range(3):
                for patch_width in range(3):
                    #check dist 
                    for annotated in self.modification_list:
                        annotated_coord = annotated['feature_vector']
                        radius = annotated['radius']
                        good_bool = annotated['good_bool']
                        if self.l2(annotated_coord,self.output_slide[i][patch_height][patch_width]) < radius:
                            self.state_img_default[3*tile_height+patch_height,3*tile_width+patch_width,:] = _color[good_bool]
                            _count +=1 
        print('patch changed by previous annotation:',_count)


    def state_img_generation(self):
        state_img = np.zeros((self.SlideRescaleHeight*3,self.SlideRescaleWidth*3,3))
        self.assignment_slide={}
        self.output_slide={}
        for i in self.output_dict:
            if i.split('/')[0] == self.SlideName:
                self.assignment_slide[i] = self.assignment[i]
                self.output_slide[i] = self.output_dict[i]
        borderline_patch_list,self.good_cluster,self.bad_cluster = borderline_sample_detection(self.M,self.output_slide,self.ClusterSelection,self.lambda_1)# [(tile_name,patch_height_ind,patch_width_ind),...]
        # self.good_cluster,self.bad_cluster = good_bad_cluster_selection(self.ClusterSelection)
        print('good cluster',self.good_cluster)
        for i in self.assignment_slide:
            if i.split('/')[0] == self.SlideName:
                # generate the state img without borderline info
                tile_height,tile_width = i.split('TileLoc_')[-1].split('.')[0].split('_')[:2]
                tile_height = int(tile_height)
                tile_width = int(tile_width)
                tile_assignment = self.assignment_slide[i]
                print(tile_assignment[0][0])
                for patch_height in range(3):
                    for patch_width in range(3):
                        if tile_assignment[patch_height][patch_width] in self.good_cluster:
                            state_img[3*tile_height+patch_height,3*tile_width+patch_width,:] = 255
                        elif tile_assignment[patch_height][patch_width] in self.bad_cluster:
                            state_img[3*tile_height+patch_height,3*tile_width+patch_width,:] = 50
                        else:
                            state_img[3*tile_height+patch_height,3*tile_width+patch_width,:] = 100
        # add borderline info
        for i in borderline_patch_list:
            tile_height,tile_width = i[0].split('TileLoc_')[-1].split('.')[0].split('_')[:2]
            tile_height = int(tile_height)
            tile_width = int(tile_width)
            patch_height,patch_width = i[1:3]
            state_img[3*tile_height+patch_height,3*tile_width+patch_width,0] = 255
            state_img[3*tile_height+patch_height,3*tile_width+patch_width,1:] = 100
        print('total good patches: ', np.sum(state_img[:,:,0] == 255))
        return state_img
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
        if event.type() == QtCore.QEvent.Wheel and 'pos' in dir(event):
            print("Enter Wheel Event")
            if event.globalX() < self.viewer2.x():
                self.viewer2.wheelEvent(event)
                self.viewer2.verticalScrollBar().setValue(self.viewer1.verticalScrollBar().value())
                self.viewer2.horizontalScrollBar().setValue(self.viewer1.horizontalScrollBar().value())
            elif event.globalX() >= self.viewer2.x():
                self.viewer1.wheelEvent(event)
                self.viewer1.verticalScrollBar().setValue(self.viewer2.verticalScrollBar().value())
                self.viewer1.horizontalScrollBar().setValue(self.viewer2.horizontalScrollBar().value())
        elif event.type() in [QtCore.QEvent.MouseButtonPress,QtCore.QEvent.MouseButtonRelease,QtCore.QEvent.ContextMenu]:
            if event.globalX() < self.viewer2.x():
                self.viewer2.verticalScrollBar().setValue(self.viewer1.verticalScrollBar().value())
                self.viewer2.horizontalScrollBar().setValue(self.viewer1.horizontalScrollBar().value())
            elif event.globalX() >= self.viewer2.x():
                self.viewer1.verticalScrollBar().setValue(self.viewer2.verticalScrollBar().value())
                self.viewer1.horizontalScrollBar().setValue(self.viewer2.horizontalScrollBar().value())
        elif event.type() == QtCore.QEvent.MouseMove:
            # print('mouse move',event.pos())

            if event.globalX() < self.viewer2.x():  
                scale_factor = self.viewer1.transform().m11() 
                transform = self.viewer1.transform()
                bar_x =    self.viewer1.horizontalScrollBar().value()          
                canvas_x = self.viewer1.horizontalScrollBar().value() + event.x()
                bar_y =    self.viewer1.verticalScrollBar().value()
                canvas_y = self.viewer1.verticalScrollBar().value() + event.y()

                self.viewer2.verticalScrollBar().setValue(self.viewer1.verticalScrollBar().value())
                self.viewer2.horizontalScrollBar().setValue(self.viewer1.horizontalScrollBar().value())

            elif event.globalX() >= self.viewer2.x():
                scale_factor = self.viewer2.transform().m11()
                transform = self.viewer2.transform()
                bar_x =    self.viewer2.horizontalScrollBar().value()          
                canvas_x = self.viewer2.horizontalScrollBar().value() + event.x()
                bar_y =    self.viewer2.verticalScrollBar().value()
                canvas_y = self.viewer2.verticalScrollBar().value() + event.y()

                self.viewer1.verticalScrollBar().setValue(self.viewer2.verticalScrollBar().value())
                self.viewer1.horizontalScrollBar().setValue(self.viewer2.horizontalScrollBar().value())
            
            if self.loaded_img_bool:
                _zoom = self.viewer1._zoom
                self.loadImage()
                self.viewer1.setTransform(transform)
                self.viewer1.verticalScrollBar().setValue(bar_y)
                self.viewer1.horizontalScrollBar().setValue(bar_x)
                self.viewer1._zoom = _zoom
                self.viewer2.setTransform(transform)
                self.viewer2.verticalScrollBar().setValue(bar_y)
                self.viewer2.horizontalScrollBar().setValue(bar_x)
                self.viewer2._zoom = _zoom

            self.actual_x = canvas_x//scale_factor
            self.actual_y = canvas_y//scale_factor
            # print(self.actual_x,self.actual_y)           
            if self.loaded_img_bool:
                # print(self.SlideRescaleHeight,self.SlideRescaleWidth)
                self.img_extract = deepcopy(patch_image_extract(self.SlideImg,self.actual_y,self.actual_x)) # (slide, h, w)
                patch_qim = ImageQt(self.img_extract)
                self.viewer3.setPhoto(QPixmap(QtGui.QPixmap.fromImage(patch_qim)))
        elif event.type() == QtCore.QEvent.MouseButtonDblClick:
            if event.button() == Qt.LeftButton:                
                self.actual_x_boubleclicked = self.actual_x
                self.actual_y_boubleclicked = self.actual_y
                self.buildExamplePopup(self.img_extract)

        return False
    def buildExamplePopup(self,img_extract):
        # name = item.text()
        exPopup = examplePopup(img_extract)
        self.exPopup = exPopup
        self.exPopup.got_annotation.connect(self.got_single_annotation)
        # self.exPopup.setGeometry(QtCore.QRect(100, 100, 600, 600))
        self.exPopup.show()

    def buildFinishPopup(self):
        FiPopup = FinishPopup()
        self.FiPopup = FiPopup
        self.FiPopup.finish_process.connect(self.finish_core)
        self.FiPopup.show()


    def got_single_annotation(self,good_bool):
        # get the location
        tile_height_num = self.actual_y_boubleclicked//3
        tile_width_num = self.actual_x_boubleclicked//3

        patch_height_num = self.actual_y_boubleclicked - tile_height_num*3
        patch_width_num = self.actual_x_boubleclicked - tile_width_num*3

        tile_name = self.tile_name_search(tile_height_num,tile_width_num)
        if tile_name:
            print(tile_name)
            # get which cluster it belongs to, calculate the dist and determine the surrounding area
            cluster_ind = self.assignment_slide[tile_name][int(patch_height_num)][int(patch_width_num)]
            feature_vector = self.output_slide[tile_name][int(patch_height_num)][int(patch_width_num)]
            dist = self.l2(feature_vector,self.M[:,cluster_ind])
            # add annotation
            # also (tile_name, patch_heigh_ind, patch_width_ind, good_bool, feature_vector, radius)
            #self.modification_list.append((tile_name,patch_height_num,patch_width_num,good_bool,feature_vector,self.lambda_2*dist))
            anno_single_dict = {
                'tile_name': tile_name,
                'patch_height_ind': patch_height_num,
                'patch_width_ind': patch_width_num,
                'good_bool': good_bool,
                'feature_vector': feature_vector,
                'radius': self.lambda_2*dist
            }
            self.modification_list.append(anno_single_dict)
            # sync state image
            _color = {True:255,False:50}
            state_img = np.array(self.state_img_default)
            for tile_image in self.output_slide:
                tile_height,tile_width = tile_image.split('TileLoc_')[-1].split('.')[0].split('_')[:2]
                tile_height = int(tile_height)
                tile_width = int(tile_width)
                feature_vectors_tmp = self.output_slide[tile_image]
                for patch_height in range(3):
                    for patch_width in range(3):
                        if self.l2(feature_vector,feature_vectors_tmp[patch_height][patch_width]) < self.lambda_2*dist:
                            state_img[3*tile_height+patch_height,3*tile_width+patch_width,:] = _color[good_bool]
            print('current number of good patch images: ', np.sum(state_img[:,:,0] == 255))
            self.IterationLabel.setText('Number of high-quality regions now: '+str(int(np.sum(state_img[:,:,0] == 255))))
            state_img = Image.fromarray(state_img.astype(np.uint8))
            self.state_img_default = deepcopy(state_img)
            qim_state = ImageQt(deepcopy(self.state_img_default))

            transform = self.viewer1.transform()
            verticalScrollBar = self.viewer1.verticalScrollBar().value()
            horizontalScrollBar = self.viewer1.horizontalScrollBar().value()

            qim = ImageQt(deepcopy(self.resized_SlideImg_default))

            self.viewer1.setPhoto(QPixmap(QtGui.QPixmap.fromImage(qim)))
            self.viewer1.setTransform(transform)
            self.viewer1.verticalScrollBar().setValue(verticalScrollBar)
            self.viewer1.horizontalScrollBar().setValue(horizontalScrollBar)

            self.viewer2.setPhoto(QPixmap(QtGui.QPixmap.fromImage(qim_state)))
            self.viewer2.setTransform(transform)
            self.viewer2.verticalScrollBar().setValue(verticalScrollBar)
            self.viewer2.horizontalScrollBar().setValue(horizontalScrollBar)

            # self.finish()

    def l2(self,point_1,point_2):
        assert len(point_1) == len(point_2)
        return np.sqrt(np.sum((np.array(point_1)-np.array(point_2))**2))
    def tile_name_search(self,tile_height_num,tile_width_num):
        for i in self.assignment_slide:
            if '_TileLoc_'+str(int(tile_height_num))+'_'+str(int(tile_width_num)) in i:
                return i
        return None

    def finish(self):
        self.buildFinishPopup()
    def finish_core(self,save_bool):
        if save_bool:
            OutputDict = {}
            OutputDict['NetworkInputPath'] = self.NetworkInputPath
            OutputDict['SlideNum'] = self.SlideName
            OutputDict['lambda_1'] = self.lambda_1
            OutputDict['lambda_2'] = self.lambda_2
            OutputDict['modification'] = self.modification_list
            with open(os.path.join(self.Output_dir,self.SlideName+'.json'),'w') as a:
                    json.dump(OutputDict,a,indent=4)
        SaPopup = SavedPopup()
        self.SaPopup = SaPopup
        self.SaPopup.show()



class examplePopup(QWidget):
    got_annotation = QtCore.pyqtSignal(bool)
    def __init__(self,img_extract):
        QWidget.__init__(self)


        self.btnGood = QtWidgets.QToolButton(self)
        self.btnGood.setText('Good')
        self.btnGood.clicked.connect(self.good)

        self.btnBad = QtWidgets.QToolButton(self)
        self.btnBad.setText('Bad')
        self.btnBad.clicked.connect(self.bad)

        self.viewer = QLabel()
        self.viewer.setPixmap(QPixmap(QtGui.QPixmap.fromImage(ImageQt(deepcopy(img_extract)))).scaled(300,300))

        VBlayout = QtWidgets.QVBoxLayout(self)
        HBlayout0 = QtWidgets.QHBoxLayout()
        HBlayout0.setAlignment(QtCore.Qt.AlignLeft)
        HBlayout0.addWidget(self.btnGood)
        HBlayout0.addWidget(self.btnBad)
        VBlayout.addLayout(HBlayout0)
        VBlayout.addWidget(self.viewer)

        

    def good(self):
        self.good_bool = True
        self.emit()

    def bad(self):
        self.good_bool = False
        self.emit()

    def emit(self):
        self.got_annotation.emit(self.good_bool)
        self.close()

class FinishPopup(QWidget):
    finish_process = QtCore.pyqtSignal(bool)
    def __init__(self):
        QWidget.__init__(self)


        self.btnGood = QtWidgets.QToolButton(self)
        self.btnGood.setText('Save')
        self.btnGood.clicked.connect(self.good)

        self.btnBad = QtWidgets.QToolButton(self)
        self.btnBad.setText('Cancel and save later')
        self.btnBad.clicked.connect(self.bad)


        VBlayout = QtWidgets.QVBoxLayout(self)
        HBlayout0 = QtWidgets.QHBoxLayout()
        HBlayout0.setAlignment(QtCore.Qt.AlignLeft)
        HBlayout0.addWidget(self.btnGood)
        HBlayout0.addWidget(self.btnBad)
        VBlayout.addLayout(HBlayout0)

        
    def good(self):
        self.finish_bool = True
        self.emit()

    def bad(self):
        self.finish_bool = False
        self.emit()

    def emit(self):
        self.finish_process.emit(self.finish_bool)
        self.close()


class SavedPopup(QWidget):
    finish_process = QtCore.pyqtSignal(bool)
    def __init__(self):
        QWidget.__init__(self)
        windowExample = QWidget()
        self.Label = QLabel(windowExample)
        self.Label.setText('Refinement successfully saved.')

        VBlayout = QtWidgets.QVBoxLayout(self)
        HBlayout0 = QtWidgets.QHBoxLayout()
        HBlayout0.setAlignment(QtCore.Qt.AlignLeft)
        HBlayout0.addWidget(self.Label)
        VBlayout.addLayout(HBlayout0)



if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.setGeometry(200, 100, 1600, 1000)
    window.show()
    sys.exit(app.exec_())