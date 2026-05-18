# -*- coding: utf-8 -*-
import time
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, \
    QMessageBox, QWidget, QHeaderView, QTableWidgetItem, QAbstractItemView, QStackedWidget, \
    QVBoxLayout, QHBoxLayout, QGridLayout, QSpacerItem, QSizePolicy, QLabel
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal, QCoreApplication, QRect
from PyQt5 import QtCore, QtWidgets, QtGui
import sys
import os
from PIL import ImageFont
from ultralytics import YOLO
sys.path.append('UIProgram')
from UIProgram.UiMain import Ui_MainWindow
import detect_tools as tools
import cv2
import Config
from UIProgram.QssLoader import QSSLoader
from UIProgram.precess_bar import ProgressBar
import numpy as np
import torch
from PyQt5.QtWidgets import QStackedWidget, QMessageBox  # 用于界面管理和消息框
import UIProgram.ui_sources_rc
from UIProgram import ui_sources_rc
import json





class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        
        # 添加一个初始化标志，用于区分初始化阶段和用户主动操作
        self._initializing = True
        
        # 批量图片浏览相关变量
        self.image_list = []  # 存储图片路径列表
        self.current_image_index = -1  # 当前显示的图片索引
        self.is_batch_mode = False  # 是否处于批量图片模式
        self.is_video_mode = False  # 是否处于视频模式
        self.video_frames = []  # 存储视频帧
        self.current_frame_index = -1  # 当前显示的视频帧索引
        
        # 检测是否有可用的GPU
        self.has_gpu = torch.cuda.is_available()
        # 默认选择GPU，如果没有GPU则选择CPU
        if self.has_gpu:
            self.device = 0  # 使用第一个GPU
        else:
            self.device = 'cpu'
        
        # 初始化参数（默认值）
        self.conf_thres = 0.25  # 默认置信度阈值
        self.iou_thres = 0.45   # 默认IOU阈值
        self.show_labels = True  # 默认显示标签
        self.show_original = False  # 默认不显示原图
            
        # 加载保存的设置（会覆盖默认值）
        self.load_settings()
        
        self.initMain()
        
        # 修复UI布局问题（重叠、剪裁等）
        self.fix_ui_layout()
        
        # 初始化导航按钮
        self.initNavigationButtons()
        
        # 加载css渲染效果
        style_file = 'UIProgram/style.css'
        qssStyleSheet = QSSLoader.read_qss_file(style_file)
        self.setStyleSheet(qssStyleSheet)

        # 设置SpinBox的范围和步长
        self.doubleSpinBox.setRange(0.0, 1.0)  # 置信度阈值范围
        self.doubleSpinBox.setSingleStep(0.05)  # 步长
        self.doubleSpinBox_2.setRange(0.0, 1.0)  # IOU阈值范围
        self.doubleSpinBox_2.setSingleStep(0.05)  # 步长
        
        # 设置SpinBox的初始值（这些值已经在load_settings中设置过，这里只是确保UI显示正确）
        self.doubleSpinBox.setValue(self.conf_thres)
        self.doubleSpinBox_2.setValue(self.iou_thres)
        self.checkBox.setChecked(self.show_labels)
        self.checkBox_2.setChecked(self.show_original)
        
        # 初始化设备选择下拉框
        self.comboBox_2.clear()
        self.comboBox_2.addItems(["GPU", "CPU"])
        # 设置当前设备选择（这些值已经在load_settings中设置过，这里只是确保UI显示正确）
        if self.device == 0 and self.has_gpu:
            self.comboBox_2.setCurrentIndex(0)  # GPU
        else:
            self.comboBox_2.setCurrentIndex(1)  # CPU
        
        # 在界面上显示当前模型路径
        self.PicBtn_2.setToolTip(f"当前模型: {Config.model_path}")
        
        # 初始化完成
        self._initializing = False
        
        # 初始化完成后再连接信号，避免在初始化过程中触发信号
        self.signalconnect()

    def signalconnect(self):
        # 基本按钮和控件信号连接
        self.PicBtn.clicked.connect(self.open_img)
        self.comboBox.activated.connect(self.combox_change)
        self.VideoBtn.clicked.connect(self.vedio_show)
        self.CapBtn.clicked.connect(self.camera_show)
        self.SaveBtn.clicked.connect(self.save_detect_video)
        self.ExitBtn.clicked.connect(self.exit_app)
        self.FilesBtn.clicked.connect(self.detact_batch_imgs)
        self.PicBtn_2.clicked.connect(self.select_model)
        
        # 添加新控件的信号连接
        self.doubleSpinBox.valueChanged.connect(self.update_conf_thres)
        self.doubleSpinBox_2.valueChanged.connect(self.update_iou_thres)
        self.checkBox.stateChanged.connect(self.update_show_labels)
        self.checkBox_2.stateChanged.connect(self.update_show_original)
        
        # 上一张/下一张按钮信号连接
        self.pushButton.clicked.connect(self.show_previous_image)
        self.pushButton_2.clicked.connect(self.show_next_image)
        
        # 设备选择下拉框信号连接 - 放在最后连接，避免初始化时触发
        self.comboBox_2.currentIndexChanged.connect(self.change_device)

    def initMain(self):
        self.show_width = 770
        self.show_height = 480

        self.org_path = None

        self.is_camera_open = False
        self.cap = None

        # 加载检测模型
        self.model = YOLO(Config.model_path, task='detect')
        self.model(np.zeros((48, 48, 3)), device=self.device)  #预先加载推理模型
        self.fontC = ImageFont.truetype("Font/platech.ttf", 25, 0)

        # 用于绘制不同颜色矩形框
        self.colors = tools.Colors()

        # 更新视频图像
        self.timer_camera = QTimer()

        # 更新检测信息表格
        # self.timer_info = QTimer()
        # 保存视频
        self.timer_save_video = QTimer()

        # 表格
        self.tableWidget.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.tableWidget.verticalHeader().setDefaultSectionSize(40)
        self.tableWidget.setColumnWidth(0, 80)  # 设置列宽
        self.tableWidget.setColumnWidth(1, 200)
        self.tableWidget.setColumnWidth(2, 150)
        self.tableWidget.setColumnWidth(3, 90)
        self.tableWidget.setColumnWidth(4, 230)
        # self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)  # 表格铺满
        # self.tableWidget.horizontalHeader().setSectionResizeMode(0, QHeaderView.Interactive)
        # self.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)  # 设置表格不可编辑
        self.tableWidget.setSelectionBehavior(QAbstractItemView.SelectRows)  # 设置表格整行选中
        self.tableWidget.verticalHeader().setVisible(False)  # 隐藏列标题
        self.tableWidget.setAlternatingRowColors(True)  # 表格背景交替

        # 设置主页背景图片border-image: url(:/icons/ui_imgs/icons/camera.png)
        # self.setStyleSheet("#MainWindow{background-image:url(:/bgs/ui_imgs/bg3.jpg)}")

    def fix_ui_layout(self):
        """修复UI布局问题，使界面自适应窗口大小"""
        # 1. 解锁窗口最大尺寸限制，使其可缩放
        self.setMaximumSize(16777215, 16777215)
        self.setMinimumSize(1000, 700)
        
        # 2. 创建主布局系统 (Central Widget Layout)
        # 将原有三个主要 Frame (标题、主显示区、右侧侧边栏) 放入主布局
        main_v_layout = QVBoxLayout(self.centralwidget)
        main_v_layout.setContentsMargins(15, 15, 15, 15)
        main_v_layout.setSpacing(10)
        
        # --- A. 顶部标题栏 (frame_5) ---
        main_v_layout.addWidget(self.frame_5)
        # 优化标题 label_3 的布局
        title_layout = QVBoxLayout(self.frame_5)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.addWidget(self.label_3)
        # 调整标题字体大小和策略，确保不被遮挡
        self.label_3.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label_3.setMinimumHeight(60)
        self.label_3.setStyleSheet("font-size: 36px; font-weight: bold; color: #2c3e50;")
        
        # --- B. 中部主要内容区 (主显示 + 侧边栏) ---
        content_h_layout = QHBoxLayout()
        main_v_layout.addLayout(content_h_layout, stretch=1)
        
        # B1. 左侧显示区 (frame: 包含图像 label_show 和表格 tableWidget)
        content_h_layout.addWidget(self.frame, stretch=7)
        left_v_layout = QVBoxLayout(self.frame)
        left_v_layout.setContentsMargins(0, 0, 0, 0)
        
        # 图像显示框 (frame_2)
        left_v_layout.addWidget(self.frame_2, stretch=2)
        display_layout = QVBoxLayout(self.frame_2)
        display_layout.setContentsMargins(0, 0, 0, 0)
        display_layout.setSpacing(6)
        display_layout.addWidget(self.label_show, stretch=1)
        self.label_show.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # 覆盖原有的固定尺寸样式
        self.label_show.setMinimumSize(100, 100) 
        self.label_show.setMaximumSize(16777215, 16777215)

        nav_layout = QHBoxLayout()
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(10)
        self.pushButton.show()
        self.pushButton_2.show()
        nav_layout.addWidget(self.pushButton)
        nav_layout.addStretch(1)
        nav_layout.addWidget(self.pushButton_2)
        display_layout.addLayout(nav_layout)

        if hasattr(self, 'PiclineEdit'):
            self.PiclineEdit.hide()
        if hasattr(self, 'VideolineEdit'):
            self.VideolineEdit.hide()
        if hasattr(self, 'CaplineEdit'):
            self.CaplineEdit.hide()
        
        # 下方表格区 (frame_3)
        left_v_layout.addWidget(self.frame_3, stretch=1)
        table_v_layout = QVBoxLayout(self.frame_3)
        table_v_layout.setContentsMargins(0, 0, 0, 0)
        table_v_layout.addWidget(self.groupBox_3)
        group_table_layout = QVBoxLayout(self.groupBox_3)
        group_table_layout.addWidget(self.tableWidget)
        
        # B2. 右侧控制侧边栏 (frame_4)
        content_h_layout.addWidget(self.frame_4, stretch=3)
        right_v_layout = QVBoxLayout(self.frame_4)
        right_v_layout.setContentsMargins(0, 0, 0, 0)
        
        # 检测参数设置 (groupBox)
        right_v_layout.addWidget(self.groupBox)
        # 检测结果 (groupBox_2)
        right_v_layout.addWidget(self.groupBox_2)
        # 操作按钮 (groupBox_4)
        right_v_layout.addWidget(self.groupBox_4)
        
        # 优化 groupBox_2 内部布局
        if hasattr(self, 'layoutWidget_2'): self.layoutWidget_2.hide()
        if hasattr(self, 'layoutWidget4'): self.layoutWidget4.hide()
        if hasattr(self, 'layoutWidget5'): self.layoutWidget5.hide()
        if hasattr(self, 'layoutWidget_3'): self.layoutWidget_3.hide()
        if hasattr(self, 'layoutWidget6'): self.layoutWidget6.hide()
        if hasattr(self, 'frame_6'): self.frame_6.hide() # 隐藏原有的坐标详情区
        
        res_grid = QGridLayout(self.groupBox_2)
        res_grid.setContentsMargins(15, 25, 15, 15)
        res_grid.setSpacing(10)
        res_grid.addWidget(QtWidgets.QLabel("用时："), 0, 0)
        res_grid.addWidget(self.time_lb, 0, 1)
        res_grid.addWidget(QtWidgets.QLabel("目标数目："), 0, 2)
        res_grid.addWidget(self.label_nums, 0, 3)
        res_grid.addWidget(QtWidgets.QLabel("类别："), 1, 0)
        res_grid.addWidget(self.type_lb, 1, 1)
        res_grid.addWidget(QtWidgets.QLabel("置信度："), 1, 2)
        res_grid.addWidget(self.label_conf, 1, 3)
        res_grid.addWidget(QtWidgets.QLabel("目标选择："), 2, 0)
        res_grid.addWidget(self.comboBox, 2, 1, 1, 3)
        
        # 优化 groupBox (参数设置) 内部布局
        if hasattr(self, 'horizontalLayoutWidget'): self.horizontalLayoutWidget.hide()
        if hasattr(self, 'horizontalLayoutWidget_2'): self.horizontalLayoutWidget_2.hide()
        if hasattr(self, 'horizontalLayoutWidget_3'): self.horizontalLayoutWidget_3.hide()
        if hasattr(self, 'layoutWidget_4'): self.layoutWidget_4.hide()
        
        param_grid = QGridLayout(self.groupBox)
        param_grid.setContentsMargins(15, 25, 15, 15)
        param_grid.setSpacing(10)
        param_grid.addWidget(self.label_14, 0, 0) # 置信度阈值标签
        param_grid.addWidget(self.doubleSpinBox, 0, 1)
        param_grid.addWidget(self.label_15, 0, 2) # IOU阈值标签
        param_grid.addWidget(self.doubleSpinBox_2, 0, 3)
        param_grid.addWidget(self.checkBox, 1, 0, 1, 2)
        param_grid.addWidget(self.checkBox_2, 1, 2, 1, 2)
        param_grid.addWidget(self.label_16, 2, 0) # 设备选择标签
        param_grid.addWidget(self.comboBox_2, 2, 1)
        param_grid.addWidget(self.label_17, 2, 2) # 模型选择标签
        param_grid.addWidget(self.PicBtn_2, 2, 3)

        # 优化 groupBox_4 内部按钮布局，使用网格布局代替固定坐标
        # 我们重新整理一下 groupBox_4 里的按钮
        # 移除原有的 layoutWidget7 干扰，直接在 groupBox_4 上创建布局
        if hasattr(self, 'layoutWidget7'):
            self.layoutWidget7.hide() # 隐藏原有的固定坐标布局容器
            
        btn_grid = QGridLayout(self.groupBox_4)
        btn_grid.setContentsMargins(15, 25, 15, 15)
        btn_grid.setSpacing(10)
        btn_grid.addWidget(self.PicBtn, 0, 0)
        btn_grid.addWidget(self.FilesBtn, 0, 1)
        btn_grid.addWidget(self.VideoBtn, 1, 0)
        btn_grid.addWidget(self.CapBtn, 1, 1)
        btn_grid.addWidget(self.SaveBtn, 2, 0)
        btn_grid.addWidget(self.ExitBtn, 2, 1)
        
        # 3. 优化表格样式
        self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tableWidget.setSelectionBehavior(QAbstractItemView.SelectRows)
        
        # 4. 解决字体自适应问题：为主要标签设置适当的字体策略
        for widget in self.findChildren(QtWidgets.QLabel):
            if widget != self.label_3: # 标题已经单独处理了
                widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
                # 限制过大的固定字体，允许一定程度的溢出控制
                if "px" in widget.styleSheet():
                    pass # 保留自定义QSS

    def resizeEvent(self, event):
        """窗口缩放事件，更新图像显示区域尺寸"""
        super().resizeEvent(event)
        # 更新图像显示的可用区域尺寸
        self.show_width = self.label_show.width()
        self.show_height = self.label_show.height()
        
        # 如果当前有正在显示的图像，需要根据新尺寸重新缩放显示
        if hasattr(self, 'draw_img') and self.draw_img is not None:
            # 获取缩放后的图片尺寸
            self.img_width, self.img_height = self.get_resize_size(self.draw_img)
            resize_cvimg = cv2.resize(self.draw_img, (self.img_width, self.img_height))
            pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
            self.label_show.setPixmap(pix_img)
            self.label_show.setAlignment(Qt.AlignCenter)

    def initNavigationButtons(self):
        """初始化导航按钮的样式和状态"""
        # 设置按钮文本提示
        self.pushButton.setToolTip("上一张/上一帧")
        self.pushButton_2.setToolTip("下一张/下一帧")
        
        # 设置按钮样式，确保在禁用状态下仍然可见
        style = """
        QPushButton:disabled {
            background-color: #cccccc;
            color: #666666;
            border: 1px solid #999999;
        }
        """
        self.pushButton.setStyleSheet(style)
        self.pushButton_2.setStyleSheet(style)
        
        # 初始状态下禁用导航按钮
        self.pushButton.setEnabled(False)
        self.pushButton_2.setEnabled(False)
        
        print("导航按钮初始化完成")

    def open_img(self):
        if self.cap:
            # 打开图片前关闭摄像头
            self.video_stop()
            self.is_camera_open = False
            self.CaplineEdit.setText('摄像头未开启')
            self.cap = None
        
        # 重置批量模式和视频模式
        self.is_batch_mode = False
        self.is_video_mode = False
        self.image_list = []
        self.current_image_index = -1
        self.video_frames = []
        self.current_frame_index = -1
        
        # 禁用导航按钮
        self.pushButton.setEnabled(False)
        self.pushButton_2.setEnabled(False)
        
        # 弹出的窗口名称：'打开图片'
        # 默认打开的目录：'./'
        # 只能打开.jpg与.gif结尾的图片文件
        # file_path, _ = QFileDialog.getOpenFileName(self.centralwidget, '打开图片', './', "Image files (*.jpg *.gif)")
        file_path, _ = QFileDialog.getOpenFileName(None, '打开图片', './', "Image files (*.jpg *.jpeg *.png *.bmp)")
        if not file_path:
            return

        self.comboBox.setDisabled(False)
        self.org_path = file_path
        self.org_img = tools.img_cvread(self.org_path)

        # 目标检测
        t1 = time.time()
        self.results = self.model(self.org_path, conf=self.conf_thres, iou=self.iou_thres, device=self.device)[0]
        t2 = time.time()
        take_time_str = '{:.3f} s'.format(t2 - t1)
        self.time_lb.setText(take_time_str)

        location_list = self.results.boxes.xyxy.tolist()
        self.location_list = [list(map(int, e)) for e in location_list]
        cls_list = self.results.boxes.cls.tolist()
        self.cls_list = [int(i) for i in cls_list]
        self.conf_list = self.results.boxes.conf.tolist()
        self.conf_list = ['%.2f %%' % (each*100) for each in self.conf_list]

        # 根据 show_original 决定显示原图还是检测结果
        if self.show_original:
            now_img = self.org_img.copy()
        else:
            now_img = self.results.plot()
            
        self.draw_img = now_img
        # 获取缩放后的图片尺寸
        self.img_width, self.img_height = self.get_resize_size(now_img)
        resize_cvimg = cv2.resize(now_img,(self.img_width, self.img_height))
        pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
        self.label_show.setPixmap(pix_img)
        self.label_show.setAlignment(Qt.AlignCenter)
        # 设置路径显示
        self.PiclineEdit.setText(self.org_path)

        # 目标数目
        target_nums = len(self.cls_list)
        self.label_nums.setText(str(target_nums))

        # 设置目标选择下拉框
        choose_list = ['全部']
        target_names = [Config.names[id]+ '_'+ str(index) for index,id in enumerate(self.cls_list)]
        # object_list = sorted(set(self.cls_list))
        # for each in object_list:
        #     choose_list.append(Config.CH_names[each])
        choose_list = choose_list + target_names

        self.comboBox.clear()
        self.comboBox.addItems(choose_list)

        if target_nums >= 1:
            self.type_lb.setText(Config.CH_names[self.cls_list[0]])
            self.label_conf.setText(str(self.conf_list[0]))
        #   默认显示第一个目标框坐标
        #   设置坐标位置值
            self.label_xmin.setText(str(self.location_list[0][0]))
            self.label_ymin.setText(str(self.location_list[0][1]))
            self.label_xmax.setText(str(self.location_list[0][2]))
            self.label_ymax.setText(str(self.location_list[0][3]))
        else:
            self.type_lb.setText('')
            self.label_conf.setText('')
            self.label_xmin.setText('')
            self.label_ymin.setText('')
            self.label_xmax.setText('')
            self.label_ymax.setText('')

        # # 删除表格所有行
        self.tableWidget.setRowCount(0)
        self.tableWidget.clearContents()
        self.tabel_info_show(self.location_list, self.cls_list, self.conf_list,path=self.org_path)
        
        # 更新导航按钮状态
        self.update_navigation_buttons()

    def detact_batch_imgs(self):
        if self.cap:
            # 打开图片前关闭摄像头
            self.video_stop()
            self.is_camera_open = False
            self.CaplineEdit.setText('摄像头未开启')
            self.cap = None
        
        # 重置批量模式和视频模式
        self.is_batch_mode = True
        self.is_video_mode = False
        self.image_list = []
        self.current_image_index = -1  # 初始化为-1，稍后会设置为0
        self.video_frames = []
        self.current_frame_index = -1
        
        # 在开始处理前，禁用导航按钮
        self.pushButton.setEnabled(False)
        self.pushButton_2.setEnabled(False)
        
        # 存储检测结果
        self.batch_results = []
        self.batch_location_lists = []
        self.batch_cls_lists = []
        self.batch_conf_lists = []
            
        directory = QFileDialog.getExistingDirectory(self,
                                                      "选取文件夹",
                                                      "./")  # 起始路径
        if not directory:
            self.is_batch_mode = False
            return
        
        self.org_path = directory
        img_suffix = ['jpg','png','jpeg','bmp']
        
        # 收集所有图片路径
        for file_name in os.listdir(directory):
            full_path = os.path.join(directory, file_name)
            if os.path.isfile(full_path) and file_name.split('.')[-1].lower() in img_suffix:
                self.image_list.append(full_path)
        
        # 如果没有找到图片，退出批量模式
        if not self.image_list:
            QMessageBox.warning(self, '警告', '所选文件夹中没有找到图片！')
            self.is_batch_mode = False
            return
        
        # 直接在状态标签上显示进度
        self.time_lb.setText("批量处理中...")
        
        try:
            # 批量处理所有图片
            total_images = len(self.image_list)
            for i, img_path in enumerate(self.image_list):
                # 更新进度信息
                self.PiclineEdit.setText(f"正在处理: {img_path} ({i+1}/{total_images})")
                QApplication.processEvents()  # 处理事件，确保UI响应
                
                # 读取图片
                img = tools.img_cvread(img_path)
                
                # 在处理之前在label_show中显示当前图片(快速预览)
                preview_img = img.copy()
                self.img_width, self.img_height = self.get_resize_size(preview_img)
                resize_cvimg = cv2.resize(preview_img, (self.img_width, self.img_height))
                pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
                self.label_show.setPixmap(pix_img)
                self.label_show.setAlignment(Qt.AlignCenter)
                QApplication.processEvents()  # 确保UI更新
                
                # 目标检测
                results = self.model(img_path, conf=self.conf_thres, iou=self.iou_thres, device=self.device)[0]
                
                # 存储结果
                self.batch_results.append(results)
                
                # 存储检测结果信息
                location_list = results.boxes.xyxy.tolist()
                location_list = [list(map(int, e)) for e in location_list]
                self.batch_location_lists.append(location_list)
                
                cls_list = results.boxes.cls.tolist()
                cls_list = [int(i) for i in cls_list]
                self.batch_cls_lists.append(cls_list)
                
                conf_list = results.boxes.conf.tolist()
                conf_list = ['%.2f %%' % (each * 100) for each in conf_list]
                self.batch_conf_lists.append(conf_list)
                
                # 在检测完成后在label_show中显示带检测结果的图片(快速预览)
                if not self.show_original:
                    detection_img = results.plot()
                    self.img_width, self.img_height = self.get_resize_size(detection_img)
                    resize_cvimg = cv2.resize(detection_img, (self.img_width, self.img_height))
                    pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
                    self.label_show.setPixmap(pix_img)
                    self.label_show.setAlignment(Qt.AlignCenter)
                    QApplication.processEvents()  # 确保UI更新
                
                # 每处理几张图片就处理一次事件，确保UI响应
                if i % 5 == 0:
                    QApplication.processEvents()
                
            # 目标数目更新
            total_targets = sum(len(cls_list) for cls_list in self.batch_cls_lists)
            self.label_nums.setText(str(total_targets))
                
        except Exception as e:
            QApplication.processEvents()  # 确保UI响应
            QMessageBox.warning(self, '警告', f'处理过程中发生错误：{str(e)}')
            self.is_batch_mode = False
            return
        
        # 显示第一张图片的检测结果
        if total_images > 0:
            self.current_image_index = 0  # 明确设置为0
            self.show_batch_result(self.current_image_index)
            
            # 更新导航按钮状态
            self.update_navigation_buttons()
            print(f"批量处理完成，当前索引: {self.current_image_index}，总图片数: {len(self.image_list)}")
        
        # 确保UI完全响应
        QApplication.processEvents()
        
        # 提示用户检测完成
        QMessageBox.information(self, '提示', f'批量检测完成！共检测 {total_images} 张图片，共检测到 {total_targets} 个目标，现在可以使用左右箭头按钮浏览检测结果。')

    def show_batch_result(self, index):
        """显示批量处理的图片检测结果"""
        if not self.is_batch_mode or index < 0 or index >= len(self.image_list):
            return
        
        # 获取当前图片路径和检测结果
        img_path = self.image_list[index]
        self.org_img = tools.img_cvread(img_path)
        self.results = self.batch_results[index]
        self.location_list = self.batch_location_lists[index]
        self.cls_list = self.batch_cls_lists[index]
        self.conf_list = self.batch_conf_lists[index]
        
        # 设置检测时间为已批量处理
        self.time_lb.setText("批量处理")
        
        # 根据 show_original 决定显示原图还是检测结果
        if self.show_original:
            now_img = self.org_img.copy()
        else:
            now_img = self.results.plot()

        self.draw_img = now_img
        # 获取缩放后的图片尺寸
        self.img_width, self.img_height = self.get_resize_size(now_img)
        resize_cvimg = cv2.resize(now_img, (self.img_width, self.img_height))
        pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
        self.label_show.setPixmap(pix_img)
        self.label_show.setAlignment(Qt.AlignCenter)
        
        # 设置路径显示
        self.PiclineEdit.setText(f"{img_path} ({index+1}/{len(self.image_list)})")

        # 目标数目
        target_nums = len(self.cls_list)
        self.label_nums.setText(str(target_nums))

        # 设置目标选择下拉框
        choose_list = ['全部']
        target_names = [Config.names[id] + '_' + str(i) for i, id in enumerate(self.cls_list)]
        choose_list = choose_list + target_names

        self.comboBox.clear()
        self.comboBox.addItems(choose_list)

        if target_nums >= 1:
            self.type_lb.setText(Config.CH_names[self.cls_list[0]])
            self.label_conf.setText(str(self.conf_list[0]))
            #   默认显示第一个目标框坐标
            #   设置坐标位置值
            self.label_xmin.setText(str(self.location_list[0][0]))
            self.label_ymin.setText(str(self.location_list[0][1]))
            self.label_xmax.setText(str(self.location_list[0][2]))
            self.label_ymax.setText(str(self.location_list[0][3]))
        else:
            self.type_lb.setText('')
            self.label_conf.setText('')
            self.label_xmin.setText('')
            self.label_ymin.setText('')
            self.label_xmax.setText('')
            self.label_ymax.setText('')

        # 更新表格信息
        self.tableWidget.setRowCount(0)
        self.tableWidget.clearContents()
        self.tabel_info_show(self.location_list, self.cls_list, self.conf_list, path=img_path)
        self.tableWidget.scrollToBottom()

    def show_previous_image(self):
        """显示上一张图片或视频帧"""
        if self.is_batch_mode and len(self.image_list) > 0:
            # 批量图片模式
            if self.current_image_index > 0:
                self.current_image_index -= 1
                print(f"显示上一张图片: {self.current_image_index+1}/{len(self.image_list)}")
                self.show_batch_result(self.current_image_index)
                # 更新按钮状态
                self.update_navigation_buttons()
            else:
                print("已经是第一张图片")
        elif self.is_video_mode and len(self.video_frames) > 0:
            # 视频模式
            if self.current_frame_index > 0:
                self.current_frame_index -= 1
                print(f"显示上一帧: {self.current_frame_index+1}/{len(self.video_frames)}")
                self.show_video_result(self.current_frame_index)
                # 更新按钮状态
                self.update_navigation_buttons()
            else:
                print("已经是第一帧")
        else:
            print("不在批量模式或视频模式下，无法导航")

    def show_next_image(self):
        """显示下一张图片或视频帧"""
        if self.is_batch_mode and len(self.image_list) > 0:
            # 批量图片模式
            if self.current_image_index < len(self.image_list) - 1:
                self.current_image_index += 1
                print(f"显示下一张图片: {self.current_image_index+1}/{len(self.image_list)}")
                self.show_batch_result(self.current_image_index)
                # 更新按钮状态
                self.update_navigation_buttons()
            else:
                print("已经是最后一张图片")
        elif self.is_video_mode and len(self.video_frames) > 0:
            # 视频模式
            if self.current_frame_index < len(self.video_frames) - 1:
                self.current_frame_index += 1
                print(f"显示下一帧: {self.current_frame_index+1}/{len(self.video_frames)}")
                self.show_video_result(self.current_frame_index)
                # 更新按钮状态
                self.update_navigation_buttons()
            else:
                print("已经是最后一帧")
        else:
            print("不在批量模式或视频模式下，无法导航")

    def update_navigation_buttons(self):
        """更新导航按钮的启用状态"""
        if self.is_batch_mode:
            # 批量图片模式
            self.pushButton.setEnabled(self.current_image_index > 0)
            self.pushButton_2.setEnabled(self.current_image_index < len(self.image_list) - 1)
            # 确保按钮可见
            self.pushButton.show()
            self.pushButton_2.show()
            print(f"批量图片模式: 上一张按钮状态: {self.pushButton.isEnabled()}, 下一张按钮状态: {self.pushButton_2.isEnabled()}")
        elif self.is_video_mode:
            # 视频模式
            self.pushButton.setEnabled(self.current_frame_index > 0)
            self.pushButton_2.setEnabled(self.current_frame_index < len(self.video_frames) - 1)
            # 确保按钮可见
            self.pushButton.show()
            self.pushButton_2.show()
            print(f"视频模式: 上一帧按钮状态: {self.pushButton.isEnabled()}, 下一帧按钮状态: {self.pushButton_2.isEnabled()}")
        else:
            # 单图片模式或其他模式，禁用导航按钮
            self.pushButton.setEnabled(False)
            self.pushButton_2.setEnabled(False)
            print("其他模式: 导航按钮已禁用")
        
        # 强制更新按钮状态
        QApplication.processEvents()

    def vedio_show(self):
        if self.is_camera_open:
            self.is_camera_open = False
            self.CaplineEdit.setText('摄像头未开启')
        
        # 重置批量模式和视频模式
        self.is_batch_mode = False
        self.is_video_mode = True
        self.image_list = []
        self.current_image_index = -1
        self.video_frames = []
        self.current_frame_index = -1  # 初始化为-1，稍后会设置为0
        
        # 在开始处理前，禁用导航按钮
        self.pushButton.setEnabled(False)
        self.pushButton_2.setEnabled(False)
        
        # 存储检测结果
        self.video_results = []
        self.video_location_lists = []
        self.video_cls_lists = []
        self.video_conf_lists = []

        video_path = self.get_video_path()
        if not video_path:
            self.is_video_mode = False
            return None
        
        # 提取视频帧
        self.extract_video_frames(video_path)
        
        # 如果成功提取了帧，开始批量处理
        if len(self.video_frames) > 0:
            # 直接在状态标签上显示进度
            self.time_lb.setText("视频处理中...")
            
            try:
                # 批量处理所有视频帧
                total_frames = len(self.video_frames)
                for i, frame in enumerate(self.video_frames):
                    # 更新进度信息
                    self.VideolineEdit.setText(f"视频帧处理进度: {i+1}/{total_frames}")
                    QApplication.processEvents()  # 处理事件，确保UI响应
                    
                    # 在处理之前在label_show中显示当前帧(快速预览)
                    preview_img = frame.copy()
                    self.img_width, self.img_height = self.get_resize_size(preview_img)
                    resize_cvimg = cv2.resize(preview_img, (self.img_width, self.img_height))
                    pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
                    self.label_show.setPixmap(pix_img)
                    self.label_show.setAlignment(Qt.AlignCenter)
                    QApplication.processEvents()  # 确保UI更新
                    
                    # 目标检测
                    results = self.model(frame, conf=self.conf_thres, iou=self.iou_thres, device=self.device)[0]
                    
                    # 存储结果
                    self.video_results.append(results)
                    
                    # 存储检测结果信息
                    location_list = results.boxes.xyxy.tolist()
                    location_list = [list(map(int, e)) for e in location_list]
                    self.video_location_lists.append(location_list)
                    
                    cls_list = results.boxes.cls.tolist()
                    cls_list = [int(i) for i in cls_list]
                    self.video_cls_lists.append(cls_list)
                    
                    conf_list = results.boxes.conf.tolist()
                    conf_list = ['%.2f %%' % (each * 100) for each in conf_list]
                    self.video_conf_lists.append(conf_list)
                    
                    # 在检测完成后在label_show中显示带检测结果的帧(快速预览)
                    if not self.show_original:
                        detection_img = results.plot()
                        self.img_width, self.img_height = self.get_resize_size(detection_img)
                        resize_cvimg = cv2.resize(detection_img, (self.img_width, self.img_height))
                        pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
                        self.label_show.setPixmap(pix_img)
                        self.label_show.setAlignment(Qt.AlignCenter)
                        QApplication.processEvents()  # 确保UI更新
                    
                    # 每处理几帧就处理一次事件，确保UI响应
                    if i % 5 == 0:
                        QApplication.processEvents()
                
                # 目标数目更新
                total_targets = sum(len(cls_list) for cls_list in self.video_cls_lists)
                self.label_nums.setText(str(total_targets))
                
            except Exception as e:
                QApplication.processEvents()  # 确保UI响应
                QMessageBox.warning(self, '警告', f'处理过程中发生错误：{str(e)}')
                self.is_video_mode = False
                return
            
            # 显示第一帧的检测结果
            if total_frames > 0:
                self.current_frame_index = 0  # 明确设置为0
                self.show_video_result(self.current_frame_index)
            
            # 更新导航按钮状态
            self.update_navigation_buttons()
            print(f"视频处理完成，当前帧索引: {self.current_frame_index}，总帧数: {len(self.video_frames)}")
        
            # 确保UI完全响应
            QApplication.processEvents()
            
            # 提示用户检测完成
            QMessageBox.information(self, '提示', f'视频检测完成！共检测 {total_frames} 帧，共检测到 {total_targets} 个目标，现在可以使用左右箭头按钮浏览检测结果。')
        else:
            # 如果未能提取帧，回退到常规视频播放模式
            self.is_video_mode = False
            self.cap = cv2.VideoCapture(video_path)
            self.video_start()
            self.comboBox.setDisabled(True)

    def extract_video_frames(self, video_path):
        """从视频中提取帧"""
        self.video_frames = []
        cap = cv2.VideoCapture(video_path)
        
        # 获取视频总帧数
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 如果帧数太多，只提取部分帧（每秒一帧）
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps)  # 每秒一帧
        if frame_interval < 1:
            frame_interval = 1
        
        # 在状态标签上显示进度
        self.VideolineEdit.setText("正在提取视频帧...")
        QApplication.processEvents()  # 确保UI响应
        
        try:
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 每10帧更新一次进度显示
                if frame_count % 10 == 0:
                    self.VideolineEdit.setText(f"正在提取视频帧: {frame_count}/{total_frames}")
                    QApplication.processEvents()  # 处理事件，确保UI响应
                
                # 如果当前帧是我们需要提取的帧
                if frame_count % frame_interval == 0:
                    # 保存帧
                    self.video_frames.append(frame.copy())
                    
                    # 在label_show中显示当前提取的帧(快速预览)
                    preview_img = frame.copy()
                    self.img_width, self.img_height = self.get_resize_size(preview_img)
                    resize_cvimg = cv2.resize(preview_img, (self.img_width, self.img_height))
                    pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
                    self.label_show.setPixmap(pix_img)
                    self.label_show.setAlignment(Qt.AlignCenter)
                    QApplication.processEvents()  # 确保UI更新
                
                frame_count += 1
                
        except Exception as e:
            QApplication.processEvents()  # 确保UI响应
            QMessageBox.warning(self, '警告', f'提取视频帧时发生错误：{str(e)}')
            self.is_video_mode = False
        finally:
            # 释放资源
            cap.release()
            QApplication.processEvents()  # 强制处理所有挂起的事件
        
        if not self.video_frames:
            QMessageBox.warning(self, '警告', '无法从视频中提取帧！')
            self.is_video_mode = False

    def show_video_result(self, index):
        """显示视频帧检测结果"""
        if not self.is_video_mode or index < 0 or index >= len(self.video_frames):
            return
        
        # 获取当前帧和检测结果
        frame = self.video_frames[index]
        self.org_img = frame.copy()
        self.results = self.video_results[index]
        self.location_list = self.video_location_lists[index]
        self.cls_list = self.video_cls_lists[index]
        self.conf_list = self.video_conf_lists[index]
        
        # 设置检测时间为已批量处理
        self.time_lb.setText("批量处理")
        
        # 根据 show_original 决定显示原图还是检测结果
        if self.show_original:
            now_img = self.org_img.copy()
        else:
            now_img = self.results.plot()

        self.draw_img = now_img
        # 获取缩放后的图片尺寸
        self.img_width, self.img_height = self.get_resize_size(now_img)
        resize_cvimg = cv2.resize(now_img, (self.img_width, self.img_height))
        pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
        self.label_show.setPixmap(pix_img)
        self.label_show.setAlignment(Qt.AlignCenter)
        
        # 设置路径显示
        self.PiclineEdit.setText(f"{self.org_path} - 帧 {index+1}/{len(self.video_frames)}")

        # 目标数目
        target_nums = len(self.cls_list)
        self.label_nums.setText(str(target_nums))

        # 设置目标选择下拉框
        choose_list = ['全部']
        target_names = [Config.names[id] + '_' + str(i) for i, id in enumerate(self.cls_list)]
        choose_list = choose_list + target_names

        self.comboBox.clear()
        self.comboBox.addItems(choose_list)

        if target_nums >= 1:
            self.type_lb.setText(Config.CH_names[self.cls_list[0]])
            self.label_conf.setText(str(self.conf_list[0]))
            #   默认显示第一个目标框坐标
            #   设置坐标位置值
            self.label_xmin.setText(str(self.location_list[0][0]))
            self.label_ymin.setText(str(self.location_list[0][1]))
            self.label_xmax.setText(str(self.location_list[0][2]))
            self.label_ymax.setText(str(self.location_list[0][3]))
        else:
            self.type_lb.setText('')
            self.label_conf.setText('')
            self.label_xmin.setText('')
            self.label_ymin.setText('')
            self.label_xmax.setText('')
            self.label_ymax.setText('')

        # 更新表格信息
        self.tableWidget.setRowCount(0)
        self.tableWidget.clearContents()
        self.tabel_info_show(self.location_list, self.cls_list, self.conf_list, path=f"{self.org_path} - 帧 {index+1}")
        self.tableWidget.scrollToBottom()

    def camera_show(self):
        self.is_camera_open = not self.is_camera_open
        
        # 重置批量模式和视频模式
        self.is_batch_mode = False
        self.is_video_mode = False
        self.image_list = []
        self.current_image_index = -1
        self.video_frames = []
        self.current_frame_index = -1
        
        if self.is_camera_open:
            self.CaplineEdit.setText('摄像头开启')
            self.cap = cv2.VideoCapture(0)
            self.video_start()
            self.comboBox.setDisabled(True)
        else:
            self.CaplineEdit.setText('摄像头未开启')
            self.label_show.setText('')
            if self.cap:
                self.cap.release()
                cv2.destroyAllWindows()
            self.label_show.clear()

    def get_resize_size(self, img):
        _img = img.copy()
        img_height, img_width , depth= _img.shape
        ratio = img_width / img_height
        if ratio >= self.show_width / self.show_height:
            self.img_width = self.show_width
            self.img_height = int(self.img_width / ratio)
        else:
            self.img_height = self.show_height
            self.img_width = int(self.img_height * ratio)
        return self.img_width, self.img_height

    def save_detect_video(self):
        if self.cap is None and not self.org_path:
            QMessageBox.about(self, '提示', '当前没有可保存信息，请先打开图片或视频！')
            return

        if self.is_camera_open:
            QMessageBox.about(self, '提示', '摄像头视频无法保存!')
            return

        if self.cap:
            res = QMessageBox.information(self, '提示', '保存视频检测结果可能需要较长时间，请确认是否继续保存？',QMessageBox.Yes | QMessageBox.No ,  QMessageBox.Yes)
            if res == QMessageBox.Yes:
                self.video_stop()
                com_text = self.comboBox.currentText()
                self.btn2Thread_object = btn2Thread(self.org_path, self.model, com_text, self.conf_thres, self.iou_thres, self.device)
                self.btn2Thread_object.start()
                self.btn2Thread_object.update_ui_signal.connect(self.update_process_bar)
            else:
                return
        else:
            if os.path.isfile(self.org_path):
                fileName = os.path.basename(self.org_path)
                name , end_name= fileName.rsplit(".",1)
                save_name = name + '_detect_result.' + end_name
                save_img_path = os.path.join(Config.save_path, save_name)
                # 保存图片
                cv2.imwrite(save_img_path, self.draw_img)
                QMessageBox.about(self, '提示', '图片保存成功!\n文件路径:{}'.format(save_img_path))
            else:
                img_suffix = ['jpg', 'png', 'jpeg', 'bmp']
                for file_name in os.listdir(self.org_path):
                    full_path = os.path.join(self.org_path, file_name)
                    if os.path.isfile(full_path) and file_name.split('.')[-1].lower() in img_suffix:
                        name, end_name = file_name.rsplit(".",1)
                        save_name = name + '_detect_result.' + end_name
                        save_img_path = os.path.join(Config.save_path, save_name)
                        results = self.model(full_path, conf=self.conf_thres, iou=self.iou_thres, device=self.device)[0]
                        now_img = results.plot()
                        # 保存图片
                        cv2.imwrite(save_img_path, now_img)

                QMessageBox.about(self, '提示', '图片保存成功!\n文件路径:{}'.format(Config.save_path))


    def update_process_bar(self,cur_num, total):
        if cur_num == 1:
            self.progress_bar = ProgressBar(self)
            self.progress_bar.show()
        if cur_num >= total:
            self.progress_bar.close()
            QMessageBox.about(self, '提示', '视频保存成功!\n文件在{}目录下'.format(Config.save_path))
            return
        if self.progress_bar.isVisible() is False:
            # 点击取消保存时，终止进程
            self.btn2Thread_object.stop()
            return
        value = int(cur_num / total *100)
        self.progress_bar.setValue(cur_num, total, value)
        QApplication.processEvents()

    # 添加新的槽函数
    def update_conf_thres(self, value):
        self.conf_thres = value
        # 更新检测参数
        if hasattr(self, 'model'):
            self.model.conf = value
            # 如果当前有图片，重新检测
            if hasattr(self, 'org_img'):
                self.detect_current_image()
        
        # 保存设置
        self.save_settings()

    def update_iou_thres(self, value):
        self.iou_thres = value
        # 更新检测参数
        if hasattr(self, 'model'):
            self.model.iou = value
            # 如果当前有图片，重新检测
            if hasattr(self, 'org_img'):
                self.detect_current_image()
        
        # 保存设置
        self.save_settings()

    def update_show_labels(self, state):
        self.show_labels = state == Qt.Checked
        # 如果当前有检测结果，重新绘制
        if hasattr(self, 'results'):
            self.draw_detection_results()
        
        # 保存设置
        self.save_settings()

    def update_show_original(self, state):
        self.show_original = state == Qt.Checked
        # 如果当前有检测结果，重新绘制
        if hasattr(self, 'results'):
            # 如果是图片模式
            if not self.cap:
                self.draw_detection_results()
            # 如果是视频或摄像头模式，不需要额外处理，因为每一帧都会重新绘制
        
        # 保存设置
        self.save_settings()

    # 添加新方法用于重新检测当前图片
    def detect_current_image(self):
        if hasattr(self, 'org_img'):
            t1 = time.time()
            self.results = self.model(self.org_img, conf=self.conf_thres, iou=self.iou_thres, device=self.device)[0]
            t2 = time.time()
            take_time_str = '{:.3f} s'.format(t2 - t1)
            self.time_lb.setText(take_time_str)

            # 更新检测结果相关信息
            location_list = self.results.boxes.xyxy.tolist()
            self.location_list = [list(map(int, e)) for e in location_list]
            cls_list = self.results.boxes.cls.tolist()
            self.cls_list = [int(i) for i in cls_list]
            self.conf_list = self.results.boxes.conf.tolist()
            self.conf_list = ['%.2f %%' % (each*100) for each in self.conf_list]

            # 更新目标数目
            target_nums = len(self.cls_list)
            self.label_nums.setText(str(target_nums))

            # 重新设置目标选择下拉框
            choose_list = ['全部']
            target_names = [Config.names[id]+ '_'+ str(index) for index,id in enumerate(self.cls_list)]
            choose_list = choose_list + target_names
            self.comboBox.clear()
            self.comboBox.addItems(choose_list)
            self.comboBox.setCurrentIndex(0)  # 设置为"全部"

            # 更新目标信息显示
            if target_nums >= 1:
                self.type_lb.setText(Config.CH_names[self.cls_list[0]])
                self.label_conf.setText(str(self.conf_list[0]))
                self.label_xmin.setText(str(self.location_list[0][0]))
                self.label_ymin.setText(str(self.location_list[0][1]))
                self.label_xmax.setText(str(self.location_list[0][2]))
                self.label_ymax.setText(str(self.location_list[0][3]))
            else:
                self.type_lb.setText('')
                self.label_conf.setText('')
                self.label_xmin.setText('')
                self.label_ymin.setText('')
                self.label_xmax.setText('')
                self.label_ymax.setText('')

            # 更新表格信息
            self.tabel_info_show(self.location_list, self.cls_list, self.conf_list, path=self.org_path)

            # 绘制检测结果
            self.draw_detection_results()

    # 添加新方法用于绘制检测结果
    def draw_detection_results(self):
        if not hasattr(self, 'results'):
            return
        
        # 根据 show_original 决定显示原图还是检测结果
        if self.show_original and hasattr(self, 'org_img'):
            now_img = self.org_img.copy()
        else:
            # 使用results.plot()作为基础图像
            now_img = self.results.plot()
            
            # 如果不显示标签，重新绘制只有框的图像
            if not self.show_labels:
                now_img = self.org_img.copy()
                for box in self.results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    color = self.colors(cls, True)
                    cv2.rectangle(now_img, (x1, y1), (x2, y2), color, 2)

        self.draw_img = now_img
        # 更新显示
        self.img_width, self.img_height = self.get_resize_size(now_img)
        resize_cvimg = cv2.resize(now_img, (self.img_width, self.img_height))
        pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
        self.label_show.setPixmap(pix_img)
        self.label_show.setAlignment(Qt.AlignCenter)

    def select_model(self):
        """
        选择模型文件并保存路径到配置文件
        """
        # 弹出文件选择对话框
        file_path, _ = QFileDialog.getOpenFileName(None, '选择模型文件', './', "Model files (*.pt *.pth)")
        if not file_path:
            return
        
        try:
            # 更新当前模型
            self.model = YOLO(file_path, task='detect')
            self.model(np.zeros((48, 48, 3)), device=self.device)  # 预先加载推理模型
            
            # 更新配置文件中的模型路径
            Config.model_path = file_path
            
            # 更新工具提示
            self.PicBtn_2.setToolTip(f"当前模型: {Config.model_path}")
            
            # 保存设置
            self.save_settings()
                
            QMessageBox.information(self, '提示', f'模型已成功加载！\n模型路径：{file_path}')
        except Exception as e:
            QMessageBox.warning(self, '警告', f'模型加载失败！\n错误信息：{str(e)}')

    def change_device(self):
        # 如果是初始化阶段，不做任何处理
        if hasattr(self, '_initializing') and self._initializing:
            return
            
        # 获取当前选择的设备
        selected_device = self.comboBox_2.currentText()
        
        # 获取之前的设备设置，用于判断是否是用户主动切换
        previous_device = "GPU" if self.device == 0 else "CPU"
        
        if selected_device == "GPU":
            if self.has_gpu:
                self.device = 0  # 使用第一个GPU
            else:
                # 只有在用户主动从CPU切换到GPU时才显示警告
                if previous_device == "CPU":
                    QMessageBox.warning(self, '警告', '没有检测到可用的GPU！\n请确认您的设备有独立显卡并已正确安装CUDA。')
                self.comboBox_2.setCurrentIndex(1)  # 切换回CPU
                self.device = 'cpu'
                return
        else:
            self.device = 'cpu'
        
        try:
            # 重新加载模型到新设备
            self.model = YOLO(Config.model_path, task='detect')
            self.model(np.zeros((48, 48, 3)), device=self.device)  # 预先加载推理模型
            
            # 更新检测参数
            self.model.conf = self.conf_thres
            self.model.iou = self.iou_thres
            
            # 显示成功消息
            device_name = "GPU" if self.device == 0 else "CPU"
            QMessageBox.information(self, '提示', f'已成功切换到{device_name}进行检测！')
            
            # 如果当前有图片，重新检测
            if hasattr(self, 'org_img'):
                self.detect_current_image()
            
            # 保存设置
            self.save_settings()
        except Exception as e:
            QMessageBox.warning(self, '警告', f'设备切换失败！\n错误信息：{str(e)}')

    def save_settings(self):
        """
        保存当前设置到配置文件
        """
        try:
            # 创建设置字典
            settings = {
                "model_path": Config.model_path,
                "device": "GPU" if self.device == 0 else "CPU",
                "conf_thres": self.conf_thres,
                "iou_thres": self.iou_thres,
                "show_labels": self.show_labels,
                "show_original": self.show_original
            }
            
            # 将设置保存到JSON文件
            settings_file = 'app_settings.json'
            with open(settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=4)
                
            print(f"设置已保存到 {settings_file}")
        except Exception as e:
            print(f"保存设置失败: {str(e)}")

    def load_settings(self):
        """
        从配置文件加载设置
        """
        settings_file = 'app_settings.json'
        if os.path.exists(settings_file):
            try:
                with open(settings_file, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                
                # 加载模型路径
                if "model_path" in settings and os.path.exists(settings["model_path"]):
                    Config.model_path = settings["model_path"]
                
                # 加载设备设置
                if "device" in settings:
                    if settings["device"] == "GPU":
                        if self.has_gpu:
                            self.device = 0
                            self.comboBox_2.setCurrentIndex(0)
                        else:
                            # 如果保存的设置是GPU但没有GPU，则使用CPU但不弹出警告
                            # 因为这是在初始化时加载，不应该打扰用户
                            self.device = 'cpu'
                            self.comboBox_2.setCurrentIndex(1)
                    else:
                        # 如果保存的设置是CPU，无论是否有GPU都使用CPU
                        self.device = 'cpu'
                        self.comboBox_2.setCurrentIndex(1)
                
                # 加载置信度阈值
                if "conf_thres" in settings:
                    self.conf_thres = float(settings["conf_thres"])
                    self.doubleSpinBox.setValue(self.conf_thres)
                
                # 加载IOU阈值
                if "iou_thres" in settings:
                    self.iou_thres = float(settings["iou_thres"])
                    self.doubleSpinBox_2.setValue(self.iou_thres)
                
                # 加载显示标签设置
                if "show_labels" in settings:
                    self.show_labels = bool(settings["show_labels"])
                    self.checkBox.setChecked(self.show_labels)
                
                # 加载显示原图设置
                if "show_original" in settings:
                    self.show_original = bool(settings["show_original"])
                    self.checkBox_2.setChecked(self.show_original)
                
                print(f"设置已从 {settings_file} 加载")
            except Exception as e:
                print(f"加载设置失败: {str(e)}")

    def exit_app(self):
        """
        退出应用程序前保存设置
        """
        self.save_settings()
        QCoreApplication.quit()

    def draw_rect_and_tabel(self, results, img):
        now_img = img.copy()
        location_list = results.boxes.xyxy.tolist()
        self.location_list = [list(map(int, e)) for e in location_list]
        cls_list = results.boxes.cls.tolist()
        self.cls_list = [int(i) for i in cls_list]
        self.conf_list = results.boxes.conf.tolist()
        self.conf_list = ['%.2f %%' % (each * 100) for each in self.conf_list]

        for loacation, type_id, conf in zip(self.location_list, self.cls_list, self.conf_list):
            type_id = int(type_id)
            color = self.colors(int(type_id), True)
            # cv2.rectangle(now_img, (int(x1), int(y1)), (int(x2), int(y2)), colors(int(type_id), True), 3)
            now_img = tools.drawRectBox(now_img, loacation, Config.CH_names[type_id], self.fontC, color)

        # 获取缩放后的图片尺寸
        self.img_width, self.img_height = self.get_resize_size(now_img)
        resize_cvimg = cv2.resize(now_img, (self.img_width, self.img_height))
        pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
        self.label_show.setPixmap(pix_img)
        self.label_show.setAlignment(Qt.AlignCenter)
        # 设置路径显示
        self.PiclineEdit.setText(self.org_path)

        # 目标数目
        target_nums = len(self.cls_list)
        self.label_nums.setText(str(target_nums))
        if target_nums >= 1:
            self.type_lb.setText(Config.CH_names[self.cls_list[0]])
            self.label_conf.setText(str(self.conf_list[0]))
            self.label_xmin.setText(str(self.location_list[0][0]))
            self.label_ymin.setText(str(self.location_list[0][1]))
            self.label_xmax.setText(str(self.location_list[0][2]))
            self.label_ymax.setText(str(self.location_list[0][3]))
        else:
            self.type_lb.setText('')
            self.label_conf.setText('')
            self.label_xmin.setText('')
            self.label_ymin.setText('')
            self.label_xmax.setText('')
            self.label_ymax.setText('')

        # 删除表格所有行
        self.tableWidget.setRowCount(0)
        self.tableWidget.clearContents()
        self.tabel_info_show(self.location_list, self.cls_list, self.conf_list, path=self.org_path)
        return now_img

    def combox_change(self):
        com_text = self.comboBox.currentText()
        if com_text == '全部':
            cur_box = self.location_list
            if not self.show_original:
                cur_img = self.results.plot()
            else:
                cur_img = self.org_img.copy()
            self.type_lb.setText(Config.CH_names[self.cls_list[0]])
            self.label_conf.setText(str(self.conf_list[0]))
        else:
            index = int(com_text.split('_')[-1])
            cur_box = [self.location_list[index]]
            if not self.show_original:
                cur_img = self.results[index].plot()
            else:
                cur_img = self.org_img.copy()
                # 如果显示原图，但需要绘制当前选中的目标框
                x1, y1, x2, y2 = self.location_list[index]
                cls = self.cls_list[index]
                color = self.colors(cls, True)
                cv2.rectangle(cur_img, (x1, y1), (x2, y2), color, 2)
                if self.show_labels:
                    label = f"{Config.CH_names[cls]} {self.conf_list[index]}"
                    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                    cv2.rectangle(cur_img, (x1, y1 - t_size[1] - 3), (x1 + t_size[0], y1), color, -1)
                    cv2.putText(cur_img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255, 255, 255], 1)
            self.type_lb.setText(Config.CH_names[self.cls_list[index]])
            self.label_conf.setText(str(self.conf_list[index]))

        # 设置坐标位置值
        self.label_xmin.setText(str(cur_box[0][0]))
        self.label_ymin.setText(str(cur_box[0][1]))
        self.label_xmax.setText(str(cur_box[0][2]))
        self.label_ymax.setText(str(cur_box[0][3]))

        resize_cvimg = cv2.resize(cur_img, (self.img_width, self.img_height))
        pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
        self.label_show.clear()
        self.label_show.setPixmap(pix_img)
        self.label_show.setAlignment(Qt.AlignCenter)

    def get_video_path(self):
        file_path, _ = QFileDialog.getOpenFileName(None, '打开视频', './', "Image files (*.avi *.mp4 *.wmv *.mkv)")
        if not file_path:
            return None
        self.org_path = file_path
        self.VideolineEdit.setText(file_path)
        return file_path

    def video_start(self):
        # 删除表格所有行
        self.tableWidget.setRowCount(0)
        self.tableWidget.clearContents()

        # 清空下拉框
        self.comboBox.clear()

        # 定时器开启，每隔一段时间，读取一帧
        self.timer_camera.start(1)
        self.timer_camera.timeout.connect(self.open_frame)

    def tabel_info_show(self, locations, clses, confs, path=None):
        path = path
        for location, cls, conf in zip(locations, clses, confs):
            row_count = self.tableWidget.rowCount()  # 返回当前行数(尾部)
            self.tableWidget.insertRow(row_count)  # 尾部插入一行
            item_id = QTableWidgetItem(str(row_count+1))  # 序号
            item_id.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)  # 设置文本居中
            item_path = QTableWidgetItem(str(path))  # 路径
            # item_path.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

            item_cls = QTableWidgetItem(str(Config.CH_names[cls]))
            item_cls.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)  # 设置文本居中

            item_conf = QTableWidgetItem(str(conf))
            item_conf.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)  # 设置文本居中

            item_location = QTableWidgetItem(str(location)) # 目标框位置
            # item_location.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)  # 设置文本居中

            self.tableWidget.setItem(row_count, 0, item_id)
            self.tableWidget.setItem(row_count, 1, item_path)
            self.tableWidget.setItem(row_count, 2, item_cls)
            self.tableWidget.setItem(row_count, 3, item_conf)
            self.tableWidget.setItem(row_count, 4, item_location)
        self.tableWidget.scrollToBottom()

    def video_stop(self):
        self.cap.release()
        self.timer_camera.stop()
        # self.timer_info.stop()

    def open_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # 保存原始帧
            self.org_img = frame.copy()
            
            # 目标检测
            t1 = time.time()
            results = self.model(frame, conf=self.conf_thres, iou=self.iou_thres, device=self.device)[0]
            t2 = time.time()
            take_time_str = '{:.3f} s'.format(t2 - t1)
            self.time_lb.setText(take_time_str)

            location_list = results.boxes.xyxy.tolist()
            self.location_list = [list(map(int, e)) for e in location_list]
            cls_list = results.boxes.cls.tolist()
            self.cls_list = [int(i) for i in cls_list]
            self.conf_list = results.boxes.conf.tolist()
            self.conf_list = ['%.2f %%' % (each * 100) for each in self.conf_list]

            # 根据 show_original 决定显示原图还是检测结果
            if self.show_original:
                now_img = self.org_img.copy()
            else:
                now_img = results.plot()

            # 获取缩放后的图片尺寸
            self.img_width, self.img_height = self.get_resize_size(now_img)
            resize_cvimg = cv2.resize(now_img, (self.img_width, self.img_height))
            pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
            self.label_show.setPixmap(pix_img)
            self.label_show.setAlignment(Qt.AlignCenter)

            # 目标数目
            target_nums = len(self.cls_list)
            self.label_nums.setText(str(target_nums))

            # 设置目标选择下拉框
            choose_list = ['全部']
            target_names = [Config.names[id] + '_' + str(index) for index, id in enumerate(self.cls_list)]
            choose_list = choose_list + target_names

            self.comboBox.clear()
            self.comboBox.addItems(choose_list)

            if target_nums >= 1:
                self.type_lb.setText(Config.CH_names[self.cls_list[0]])
                self.label_conf.setText(str(self.conf_list[0]))
                #   默认显示第一个目标框坐标
                #   设置坐标位置值
                self.label_xmin.setText(str(self.location_list[0][0]))
                self.label_ymin.setText(str(self.location_list[0][1]))
                self.label_xmax.setText(str(self.location_list[0][2]))
                self.label_ymax.setText(str(self.location_list[0][3]))
            else:
                self.type_lb.setText('')
                self.label_conf.setText('')
                self.label_xmin.setText('')
                self.label_ymin.setText('')
                self.label_xmax.setText('')
                self.label_ymax.setText('')

            # 更新表格信息
            self.tabel_info_show(self.location_list, self.cls_list, self.conf_list, path=self.org_path)

        else:
            self.cap.release()
            self.timer_camera.stop()


class btn2Thread(QThread):
    """
    进行检测后的视频保存
    """
    # 声明一个信号
    update_ui_signal = pyqtSignal(int,int)

    def __init__(self, path, model, com_text, conf, iou, device):
        super(btn2Thread, self).__init__()
        self.org_path = path
        self.model = model
        self.com_text = com_text
        self.conf = conf
        self.iou = iou
        self.device = device
        # 用于绘制不同颜色矩形框
        self.colors = tools.Colors()
        self.is_running = True  # 标志位，表示线程是否正在运行

    def run(self):
        # VideoCapture方法是cv2库提供的读取视频方法
        cap = cv2.VideoCapture(self.org_path)
        # 设置需要保存视频的格式"xvid"
        # 该参数是MPEG-4编码类型，文件名后缀为.avi
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # 设置视频帧频
        fps = cap.get(cv2.CAP_PROP_FPS)
        # 设置视频大小
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        # VideoWriter方法是cv2库提供的保存视频方法
        # 按照设置的格式来out输出
        fileName = os.path.basename(self.org_path)
        name, end_name = fileName.split('.')
        save_name = name + '_detect_result.avi'
        save_video_path = os.path.join(Config.save_path, save_name)
        out = cv2.VideoWriter(save_video_path, fourcc, fps, size)

        prop = cv2.CAP_PROP_FRAME_COUNT
        total = int(cap.get(prop))
        print("[INFO] 视频总帧数：{}".format(total))
        cur_num = 0

        # 确定视频打开并循环读取
        while (cap.isOpened() and self.is_running):
            cur_num += 1
            print('当前第{}帧，总帧数{}'.format(cur_num, total))
            # 逐帧读取，ret返回布尔值
            # 参数ret为True 或者False,代表有没有读取到图片
            # frame表示截取到一帧的图片
            ret, frame = cap.read()
            if ret == True:
                # 检测，使用指定设备
                results = self.model(frame, conf=self.conf, iou=self.iou, device=self.device)[0]
                frame = results.plot()
                out.write(frame)
                self.update_ui_signal.emit(cur_num, total)
            else:
                break
        # 释放资源
        cap.release()
        out.release()

    def stop(self):
        self.is_running = False


if __name__ == "__main__":
    # 针对 PyQt 5.15.2 版本优化：启用高分屏（DPI）自适应缩放
    # 这能解决在 4K 屏或 Windows 系统缩放（如 150%）下字体模糊或界面错位的问题
    QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QCoreApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    
    app = QApplication(sys.argv)
    
    # 设置全局字体为微软雅黑，这是 Windows 上兼容性最好的字体，能解决部分乱码和剪裁问题
    font = QtGui.QFont("Microsoft YaHei")
    app.setFont(font)
    
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
