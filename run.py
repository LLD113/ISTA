# coding=UTF-8
import matplotlib.pyplot as plt
from PySide2 import QtCore
from PySide2.QtUiTools import QUiLoader
from PySide2.QtGui import QIcon
from PySide2.QtWidgets import QWidget, QPushButton, QVBoxLayout, QFileDialog, QApplication, QLineEdit, QTreeWidgetItem, \
    QListWidget, QTreeWidget
from matplotlib.font_manager import FontProperties
from numpy import argsort

from utils_of_all import *

from generate_testcase import *
import repair_list
from deepconcolic.main import *
from fuzzing.run_experiment import *
from calculate_robustness import *
from generate_text_data import *
from optimization import *
os.environ['QT_MAC_WANTS_LAYER'] = '1'

# The font format is set to New Rome
font_song = FontProperties(fname=r"c:\windows\fonts\Times New Roman")


class Main:

    def __init__(self):
        self.model = None  # model
        self.test = None  #
        self.label = None  # label
        self.label_path = None  # Label file path, that is.csv file path
        self.pre_label = None
        self.meta_path = None  # Metamorphosis test case save path
        self.search_ui = QUiLoader().load('search.ui')
        self.search_ui.search_listWidget.itemDoubleClicked.connect(self.selectProject)

        # -----
        self.neuron_Cover = []  # self.neuron_Cover[0] Coverage of neurons self.neuron_Cover[1] Matrix of activation
        self.KMN_Cov = None  # K
        self.NB_Cov = None  # Neuronal boundary coverage
        self.SNA_Cov = None  # Strong neuronal activation coverage
        self.TKN_Cov = None  # top-k Coverage of neurons
        self.min_max_file = None  # The neuron outputs the file path of the upper and lower boundaries
        self.location_path = None  # Location address
        # ------
        self.QTextEdit = None
        self.textEdit = None
        self.line_edit = None
        self.result_path = None
        self.ui = QUiLoader().load('main_generate.ui')
        self.ui.pushButton.clicked.connect(self.importModel)  # Click the Import Model button pushButton, call importModel function
        self.ui.pushButton_2.clicked.connect(self.loadModel)  # Click the Load Model button pushButton_2, call loadModel
        self.ui.pushButton_3.clicked.connect(self.getData)  # Click the Get Data Set button pushButton_3, call getDat function
        self.ui.pushButton_4.clicked.connect(self.getLabel)  # Click the Get Label File button pushButton_4, call getLabel function
        self.ui.pushButton_5.clicked.connect(self.dataLoad)
        # self.ui.pushButton_6.clicked.connect(self.getData2)

        # Data sets -- textual data
        self.ui.pushButton_6.clicked.connect(self.get_text_data)
        self.ui.pushButton_10.clicked.connect(self.load_text_data)

        # Text data generation
        self.ui.textButton.clicked.connect(self.get_text_SaveSrc)  #

        # search
        self.ui.action_search.triggered.connect(self.openSearch)
        self.search_ui.searchButton.clicked.connect(self.searchProject)

        # Project Document Management
        self.ui.actionopen.triggered.connect(self.openProject)
        # self.ui.setCentralWidget(self.ui.treeWidget)
        self.ui.treeWidget.itemDoubleClicked.connect(self.openFile)

        # ----- Confirmation key of coverage analysis interface; Can be modified to save without executing this method; Wait until the new image is generated to execute
        self.ui.pushButton_7.clicked.connect(self.saveCover_setting)  #
        self.ui.pushButton_16.clicked.connect(self.seedcoverage)
        self.ui.seed_text_pushButton.clicked.connect(self.seed_text_coverage)
        self.ui.pushButton_17.clicked.connect(self.show_seed_coverage)

        self.ui.pushButton_8.clicked.connect(self.getMin_MaxPath)
        self.ui.pushButton_14.clicked.connect(self.generateMin_Max)

        self.ui.createMinmax.clicked.connect(self.generateMin_Max_tf)

        # -----
        # self.ui.pushButton_9.clicked.connect(self.getMetaPath)
        # self.ui.pushButton_10.clicked.connect(self.generateMeta)
        self.ui.pushButton_11.clicked.connect(self.getsSpectrumPath)
        self.ui.pushButton_12.clicked.connect(self.generate_spectrum)
        self.ui.pushButton_13.clicked.connect(self.errorLocated)
        # self.ui.browseButton.clicked.connect(self.getParSrc)

        # DataSet
        self.ui.Minist_radioButton.clicked.connect(self.getData_MINIST)
        self.ui.Ciafr_radioButton.clicked.connect(self.getData_CIFAR10)
        self.ui.FashionMinist_radioButton.clicked.connect(self.getData_fashionMINIST)
        self.ui.self_defining.clicked.connect(self.getData_self_defining)

        # Generate Settings -- Image data
        self.ui.genSetButton.clicked.connect(self.getGenSet)
        self.ui.genlocalButton_2.clicked.connect(self.getSaveSrc)
        self.ui.generate_coverage_Button.clicked.connect(self.generate_coverage)
        self.ui.generate_coverage_compare_pushButton.clicked.connect(self.generate_notext_coverage_compare)
        self.ui.ResultButton.clicked.connect(self.OpenResult)

        # Generate -- text data
        self.ui.text_gen_Button.clicked.connect(self.generate_text_data)
        self.ui.text_coverage_Button.clicked.connect(self.generate_text_coverage)
        self.ui.text_coverage_compare_pushButton.clicked.connect(self.generate_text_coverage_compare)

        # To optimize the
        self.ui.opti_openButton.clicked.connect(self.getOptimizePath)
        self.ui.optimizeButton.clicked.connect(self.openOptimize)
        self.ui.optimize_coverage_pushButton.clicked.connect(self.OptimizeCoverage)
        self.ui.compare_coverage_pushButton.clicked.connect(self.OptimizeCoverageCompare)

        # Robust controls
        self.ui.model_robustness.clicked.connect(self.import_model_robustness)
        self.ui.repaired_model_rubustness.clicked.connect(self.import_repaired_model_robustness)
        self.ui.dataset_robustness.clicked.connect(self.import_dataset_robustness)
        self.ui.dataset_label_robustness.clicked.connect(self.import_dataset_label_robustness)
        self.ui.countermeasure_path.clicked.connect(self.choose_countermeasure_save_path)
        self.ui.generate_countermeasure.clicked.connect(self.generate_countermeasure)
        self.ui.calculate_robustness.clicked.connect(self.calculate_robustness)

        # repair
        self.ui.repair_1.clicked.connect(self.error_repair_1)
        self.ui.repair_2.clicked.connect(self.error_repair_2)
        # self.ui.repair_3.clicked.connect(self.easy_set_all)
        self.model_path = None
        self.model_path_new = None
        self.model_new_repair = None
        self.ui.repair_save_path.clicked.connect(self.select_model_path_new)
        self.ui.repair_save.clicked.connect(self.save_model_new)
        self.ui.repair_result.clicked.connect(self.select_result_path)

        # self.ui.repair_1.clicked.connect(self.select_location_path)

        # Add build option control
        self.ui.gen_radioButton.clicked.connect(self.selectGenFunc)
        self.ui.cover_radioButton.clicked.connect(self.selectCoverFunc)

        # Coverage link
        self.ui.cover_1.clicked.connect(self.connect_ncover)
        self.ui.cover_2.clicked.connect(self.connect_kmncover)
        self.ui.cover_3.clicked.connect(self.connect_nbcover)
        self.ui.cover_4.clicked.connect(self.connect_snacover)
        self.ui.cover_5.clicked.connect(self.connect_tkncover)

        # Transformation of model
        self.ui.pushButton_importCkpt.clicked.connect(self.importCkpt)
        self.ui.pushButton_inCode.clicked.connect(self.inCode)
        # self.ui.pushButton_selectCovModel.clicked.connect(self.inCode)
        self.ui.pushButton_startCov.clicked.connect(self.startCov)
        self.ckptPath = None

    def startCov(self):
        os.system("python inCode.py" + " --path " + self.ckptPath)
        self.ui.logtextBrowser.append("Successfully transform the model...")
        self.ui.logtextBrowser.append("*" * 50)

    def inCode(self):
        fileName = "inCode.py"
        fp = open(fileName, 'w')
        os.startfile(fileName)

    def importCkpt(self):
        selected_file = QFileDialog.getOpenFileName()
        self.ckptPath = selected_file[0]
        # self.model_path = selected_file[0]
        self.ui.lineEdit_5.setText(self.ckptPath)

    def selectProject(self):
        items = self.search_ui.search_listWidget.selectedItems()
        project_name = items[0].text()
        self.project_path = self.project_dic[project_name].rstrip("\n")
        print(self.project_path)
        self.root = QTreeWidgetItem(self.ui.treeWidget)

        self.root.setText(0, self.project_path)
        self.cover_path = self.project_path
        self.meta_path_temp = self.project_path + '/meta testcase'
        self.generate_save_path = self.project_path + '/generate testcase'
        self.optimize_path = self.project_path + '/optimize testcase'
        self.robustness_path = self.project_path + '/robustness'
        for file in sorted(os.listdir(self.project_path)):
            QTreeWidgetItem(self.root).setText(0, file)

    # Open the search window
    def openSearch(self):

        self.search_ui.show()
        self.project_dic = {}
        with open("project_list.txt") as projects:
            for project in projects:
                pathf = os.path.dirname(project)
                project_name = project[len(pathf) + 1:]
                self.project_dic[project_name] = project
                self.search_ui.search_listWidget.addItem(project_name)

    # Search Item
    def searchProject(self):
        self.search_ui.search_listWidget.clear()
        project_name = self.search_ui.searchEdit.text()
        # count = self.search_ui.search_listWidget.count()
        for key in self.project_dic.keys():
            if project_name in key:
                self.search_ui.search_listWidget.addItem(key)

    # Open a new project
    def openProject(self):
        self.project_path = QFileDialog.getExistingDirectory()
        flag = True
        with open("project_list.txt", encoding='gbk') as projects:
            for project in projects:
                if (project == self.project_path + "\n"):
                    flag = False
        if flag:
            with open("project_list.txt", "a") as projects:
                projects.write(self.project_path + "\n")
        self.root = QTreeWidgetItem(self.ui.treeWidget)
        self.root.setText(0, self.project_path)
        self.cover_path = self.project_path
        if not os.path.exists(self.project_path + '/meta testcase'):
            os.makedirs(self.project_path + '/meta testcase')

        if not os.path.exists(self.project_path + '/repaired model'):
            os.makedirs(self.project_path + '/repaired model')

        self.model_path_new = self.project_path + '/repaired model'

        self.meta_path_temp = self.project_path + '/meta testcase'

        if not os.path.exists(self.project_path + '/generate testcase'):
            os.makedirs(self.project_path + '/generate testcase')
        self.generate_save_path = self.project_path + '/generate testcase'
        if not os.path.exists(self.project_path + '/optimize testcase'):
            os.makedirs(self.project_path + '/optimize testcase')
        self.optimize_path = self.project_path + '/optimize testcase'

        if not os.path.exists(self.project_path + '/robustness'):
            os.makedirs(self.project_path + '/robustness')
        self.robustness_path = self.project_path + '/robustness'

        for file in sorted(os.listdir(self.project_path)):
            QTreeWidgetItem(self.root).setText(0, file)
        # count = self.ui.treeWidget.columnCount()
        # # print(count)
        # # for i in range(count):
        # #     print(self.ui.treeWidget.items(i).text(0))

    def inChild(self, root, file_name):
        count = root.childCount()
        flag = True
        for i in range(count):
            if file_name == root.child(i).text(0):
                flag = False
        return flag

        # self.ui.lineEdit_7.setText(self.spectrum_path)

    def openFile(self):
        item = self.ui.treeWidget.currentItem()
        print(item)
        if item.parent() is not None:
            self.project_path = item.parent().text(0)
            os.startfile(self.project_path + "/" + item.text(0))
        else:
            self.root = item
            self.project_path = item.text(0)
            print(self.project_path)

    # 9.1 Add -------- coverage link
    def connect_ncover(self):
        self.ui.NC_checkBox_2.setChecked(self.ui.cover_1.isChecked())
        self.ui.NC_comboBox.setEnabled(self.ui.cover_1.isChecked())

    def connect_kmncover(self):
        self.ui.KMNC_checkBox_2.setChecked(self.ui.cover_2.isChecked())
        self.ui.KMNC_comboBox.setEnabled(self.ui.cover_2.isChecked())

    def connect_nbcover(self):
        self.ui.cover_4.setChecked(self.ui.cover_3.isChecked())
        self.ui.NBC_checkBox_2.setChecked(self.ui.cover_3.isChecked())
        self.ui.NBC_comboBox.setEnabled(self.ui.cover_3.isChecked())
        self.ui.SNAC_checkBox_2.setChecked(self.ui.cover_4.isChecked())
        self.ui.SNAC_comboBox.setEnabled(self.ui.cover_4.isChecked())

    def connect_snacover(self):
        self.ui.cover_3.setChecked(self.ui.cover_4.isChecked())
        self.ui.SNAC_checkBox_2.setChecked(self.ui.cover_4.isChecked())
        self.ui.SNAC_comboBox.setEnabled(self.ui.cover_4.isChecked())
        self.ui.NBC_checkBox_2.setChecked(self.ui.cover_3.isChecked())
        self.ui.NBC_comboBox.setEnabled(self.ui.cover_3.isChecked())

    def connect_tkncover(self):
        self.ui.TKNC_checkBox_2.setChecked(self.ui.cover_5.isChecked())
        self.ui.TKNC_comboBox.setEnabled(self.ui.cover_5.isChecked())

    # 9.01 Plus ------ Select the generation method and coverage
    def selectGenFunc(self):
        if (self.ui.gen_radioButton.isChecked() == True and self.ui.cover_radioButton.isChecked() == False):
            self.ui.changeSpinBox_2.setEnabled(True)
            self.ui.timeSpinBox_2.setEnabled(True)
            self.ui.numSpinBox_2.setEnabled(True)

            self.ui.NC_comboBox.setEnabled(False)
            self.ui.KMNC_comboBox.setEnabled(False)
            self.ui.NBC_comboBox.setEnabled(False)
            self.ui.SNAC_comboBox.setEnabled(False)
            self.ui.TKNC_comboBox.setEnabled(False)

            self.ui.NC_checkBox_2.setEnabled(False)
            self.ui.KMNC_checkBox_2.setEnabled(False)
            self.ui.NBC_checkBox_2.setEnabled(False)
            self.ui.SNAC_checkBox_2.setEnabled(False)
            self.ui.TKNC_checkBox_2.setEnabled(False)

    # 9.01 Plus ------ Select the generation method and coverage
    def selectCoverFunc(self):
        if (self.ui.cover_radioButton.isChecked() == True and self.ui.gen_radioButton.isChecked() == False):
            self.ui.changeSpinBox_2.setEnabled(False)
            self.ui.timeSpinBox_2.setEnabled(False)
            self.ui.numSpinBox_2.setEnabled(False)

            self.ui.NC_comboBox.setEnabled(self.ui.cover_1.isChecked())
            self.ui.NC_checkBox_2.setEnabled(self.ui.cover_1.isChecked())

            self.ui.KMNC_comboBox.setEnabled(self.ui.cover_2.isChecked())
            self.ui.KMNC_checkBox_2.setEnabled(self.ui.cover_2.isChecked())

            self.ui.NBC_comboBox.setEnabled(self.ui.cover_3.isChecked())
            self.ui.NBC_checkBox_2.setEnabled(self.ui.cover_3.isChecked())

            self.ui.SNAC_comboBox.setEnabled(self.ui.cover_4.isChecked())
            self.ui.SNAC_checkBox_2.setEnabled(self.ui.cover_4.isChecked())

            self.ui.TKNC_comboBox.setEnabled(self.ui.cover_5.isChecked())
            self.ui.TKNC_checkBox_2.setEnabled(self.ui.cover_5.isChecked())

    def radio_connect(self):
        if self.ui.self_defining.isChecked() == True:
            self.ui.lineEdit_2.setEnabled(self.ui.self_defining.isChecked())
            self.ui.pushButton_3.setEnabled(self.ui.self_defining.isChecked())
            self.ui.lineEdit_3.setEnabled(self.ui.self_defining.isChecked())
            self.ui.pushButton_4.setEnabled(self.ui.self_defining.isChecked())
        else:
            self.ui.lineEdit_2.setEnabled(self.ui.self_defining.isChecked())
            self.ui.pushButton_3.setEnabled(self.ui.self_defining.isChecked())
            self.ui.lineEdit_3.setEnabled(self.ui.self_defining.isChecked())
            self.ui.pushButton_4.setEnabled(self.ui.self_defining.isChecked())

    # Tagged MINIST data sets
    def getData_MINIST(self):
        self.radio_connect()

    def load_minist_data(self):
        img_shape = 28, 28, 1
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        x_train = x_train.reshape(x_train.shape[0], *img_shape).astype('float32') / 255
        x_test = x_test.reshape(x_test.shape[0], *img_shape).astype('float32') / 255
        # Here img_shape represents dimensions, and the others represent training and test sets, classes, and tags
        # print(x_test.shape)
        # print(len(x_test))
        # exit(0)
        # k = round(0.1 * len(x_train))
        # x_train = x_train[:k]
        return x_test

    # Labeled CIFAR10 data set
    def getData_CIFAR10(self):
        self.radio_connect()
        # self.test_path = "cifar10"

    # CIFAR10
    def load_cifar10_data(self):
        import tensorflow as tf
        img_shape = 32, 32, 3
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        print("x_test的shape是：", x_test.shape[0])
        x_train = x_train.reshape(x_train.shape[0], *img_shape).astype('float32') / 255
        x_test = x_test.reshape(x_test.shape[0], *img_shape).astype('float32') / 255
        return x_test

    # Labeled fashionMINIST data set
    def getData_fashionMINIST(self):
        self.radio_connect()
        # self.test_path = "fashion-mnist"

    # Fashion-MNIST
    def load_fashion_mnist_data(self):
        import tensorflow as tf
        img_shape = 28, 28, 1
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        x_train = x_train.reshape(x_train.shape[0], *img_shape).astype('float32') / 255
        x_test = x_test.reshape(x_test.shape[0], *img_shape).astype('float32') / 255
        # Here img_shape represents dimensions, and the others represent training and test sets, classes, and tags
        # print(x_test.shape)
        # print(len(x_test))
        # exit(0)
        # k = round(0.1 * len(x_train))
        # x_train = x_train[:k]
        return x_test

    # A custom data set with labels
    def getData_self_defining(self):
        self.radio_connect()

    # Model of selection
    def importModel(self):
        selected_file = QFileDialog.getOpenFileName()
        self.model_path = selected_file[0]
        self.ui.lineEdit.setText(self.model_path)
        # QTreeWidgetItem(self.root).setText(0, self.model_path)

    # Loading model
    def loadModel(self):
        self.model = tf.keras.models.load_model(self.model_path)
        self.model.summary()
        self.ui.logtextBrowser.append("Successfully loading the model：" + self.model_path)
        self.ui.logtextBrowser.append("*" * 50)
        trainableLayers = get_trainable_layers(self.model)
        str1 = "Trainable layer：\n"
        for l in trainableLayers:
            weights = self.model.layers[int(l)].get_weights()
            str1 += "no." + str(l + 1) + "layer，Weight shape information：" + str(weights[0].shape) + "\n"
        self.ui.textBrowser.setText(str1)
        self.ui.model_lineEdit.setText(self.model_path.split("/")[-1])

    # Unlabeled data sets
    def getData2(self):
        self.test_path = QFileDialog.getExistingDirectory()
        self.ui.lineEdit_4.setText(self.test_path)
        # QTreeWidgetItem(self.root).setText(0, self.test_path)
        self.ui.logtextBrowser.append("The data set was successfully imported. Procedure：" + self.test_path)
        self.ui.logtextBrowser.append("*" * 50)

    # Import data set
    def dataLoad(self):

        if self.ui.self_defining.isChecked() == True:
            self.ui.logtextBrowser.append("The data set was successfully imported. Procedure：" + self.test_path)
            self.ui.logtextBrowser.append("*" * 50)
            self.ui.dataset_lineEdit.setText(self.test_path.split("/")[-1])
            # self.ui.dataset_lineEdit.setText(self.test_path.split("/")[-1])
        elif self.ui.Minist_radioButton.isChecked() == True:
            self.ui.logtextBrowser.append("The minist dataset was selected successfully......")
            self.ui.logtextBrowser.append("*" * 50)
            self.ui.dataset_lineEdit.setText("minist")

        elif self.ui.Ciafr_radioButton.isChecked() == True:
            self.ui.logtextBrowser.append("The cifar-10 dataset was selected successfully......")
            self.ui.logtextBrowser.append("*" * 50)
            self.ui.dataset_lineEdit.setText("cifar-10")

        elif self.ui.FashionMinist_radioButton.isChecked() == True:
            self.ui.logtextBrowser.append("The fashion-minist dataset was successfully selected......")
            self.ui.logtextBrowser.append("*" * 50)
            self.ui.dataset_lineEdit.setText("fashion-minist")

    # Import text data set confirmation button
    def load_text_data(self):
        self.ui.logtextBrowser.append("The data set was selected successfully......")
        self.ui.logtextBrowser.append("*" * 50)
        self.ui.dataset_lineEdit.setText(self.test_path.split("/")[-1])

    # Gets the label for the imported dataset
    def getLabel(self):
        selected_file = QFileDialog.getOpenFileName(self.ui, "Select the data set label", ".", "CSV(逗号分隔)(*.csv)")
        self.label_path = selected_file[0]
        self.ui.lineEdit_3.setText(self.label_path)
        # QTreeWidgetItem(self.root).setText(0, self.label_path)

    # Labeled data sets
    def getData(self):
        self.test_path = QFileDialog.getExistingDirectory()
        self.ui.lineEdit_2.setText(self.test_path)
        # QTreeWidgetItem(self.root).setText(0, self.test_path)

    # Text data set
    def get_text_data(self):
        selected_text_file = QFileDialog.getOpenFileName(self.ui, "Select data set", ".", "CSV(逗号分隔)(*.csv)")
        self.test_path = selected_text_file[0]
        self.ui.lineEdit_4.setText(self.test_path)

    # The saved path of the restored model
    def select_model_path_new(self):
        self.ui.save_model_lineEdit.setText(self.model_path_new)

    # Save the restored model
    def save_model_new(self):

        self.model_new_repair.save(self.model_path_new + "/rambo_new.h5")
        # QTreeWidgetItem(self.root).setText(0, self.model_path_new + "/rambo_new.h5")
        self.ui.logtextBrowser.append("The restored model was successfully saved......")
        self.ui.logtextBrowser.append("*" * 50)

    # Import the sequence results of suspicious neurons
    def error_repair_1(self):
        # selected_file = QFileDialog.getOpenFileName()
        # self.suspicious_neurons_path = selected_file[0]
        self.suspicious_neurons_path = self.location_path

        self.ui.import_lineEdit.setText(self.suspicious_neurons_path)
        # self.ui.import_lineEdit.setText(self.suspicious_neurons_path)
        QTreeWidgetItem(self.root).setText(0, self.suspicious_neurons_path)
        self.ui.logtextBrowser.append("The sequence result of suspicious neurons was successfully imported......")
        self.ui.logtextBrowser.append("*" * 50)

    # Path for saving the repair result
    def select_result_path(self):

        self.result_path = self.project_path
        self.ui.repair_result_location.setText(self.result_path)

    # Repair model
    def error_repair_2(self):
        error = np.loadtxt(self.suspicious_neurons_path)
        # print(error.shape)
        # layer_number = int(self.model_path[-5])

        length = len(error)
        self.test = generate_data(self.test_path) / 255.0
        self.label = generate_label(self.label_path)
        if self.ui.checkBox_CW.isChecked():
            for i in range(0, int(length / 2)):
                n_num = error[i, 1]
                n_layer = error[i, 0]
                rm = repair_list.RepairModel(self.model, self.test, self.label, n_layer, n_num)
            model_new = rm.repair_cw()
        if self.ui.checkBox_CB.isChecked():
            for i in range(0, int(length / 2)):
                n_num = error[i, 1]
                n_layer = error[i, 0]
                rm = repair_list.RepairModel(self.model, self.test, self.label, n_layer, n_num)
            model_new = rm.repair_cb(0.2)
        if self.ui.checkBox_RAF.isChecked():
            for i in range(0, int(length / 2)):
                n_num = error[i, 1]
                n_layer = error[i, 0]
                rm = repair_list.RepairModel(self.model, self.test, self.label, n_layer, n_num)
            model_new = rm.repair_RAF()
        if self.ui.checkBox_DAF.isChecked():
            for i in range(0, int(length / 2)):
                n_num = error[i, 1]
                n_layer = error[i, 0]
                rm = repair_list.RepairModel(self.model, self.test, self.label, n_layer, n_num)
            model_new = rm.repair_DAF()
        if self.ui.checkBox_DN.isChecked():
            for i in range(0, int(length / 2)):
                n_num = error[i, 1]
                n_layer = error[i, 0]
                rm = repair_list.RepairModel(self.model, self.test, self.label, n_layer, n_num)
            model_new = rm.repair_DN()
        if self.ui.checkBox_IN.isChecked():
            for i in range(0, int(length / 2)):
                n_num = error[i, 1]
                n_layer = error[i, 0]
                rm = repair_list.RepairModel(self.model, self.test, self.label, n_layer, n_num)
            model_new = rm.repair_in()

        before_accuracy, after_accuracy = rm.repair_accuracy(model_new)
        self.ui.logtextBrowser.append("Before the restoration was：" + format(before_accuracy, '.3%'))
        self.ui.logtextBrowser.append("After restoration is：" + format(after_accuracy, '.3%'))
        if (after_accuracy * 100 - before_accuracy * 100) > 1:
            self.ui.logtextBrowser.append("Successful repair！")
            self.ui.logtextBrowser.append("*" * 50)
        else:
            self.ui.logtextBrowser.append("Repair failed. Please select another repair operator.")
            self.ui.logtextBrowser.append("*" * 50)
        # Display repair results
        rm.compare_repair(self.model, model_new, 2 / 25, self.result_path + "/result.txt")
        self.model_new_repair = model_new
        # QTreeWidgetItem(self.root).setText(0, self.model_new_repair)

    # Preloading model
    def easy_set_all(self):
        try:
            self.model = tf.keras.models.load_model("G:\Program Files\gitee3\\ai-test\\rambo.h5")
            self.test = generateData("F:\AItest\\ai-test-master\\ai-test-master\DeepTest\hmb3_small") / 255.0
            self.label = generate_label(
                "F:\AItest\\ai-test-master\\ai-test-master\DeepTest\hmb3_small\hmb3_steering.csv")
        except:
            self.ui.logtextBrowser.append("Preload failed. Model was imported from scratch")
            self.ui.logtextBrowser.append("*" * 50)
        else:
            self.ui.logtextBrowser.append("Preload successful!")
            self.ui.logtextBrowser.append("*" * 50)

    # Select the path for saving coverage
    def getCoverPath(self):
        self.cover_path = self.project_path
        self.ui.lineEdit_12.setText(self.cover_path)

    # Location Result
    def getSus(self, spectrum):
        global location, suspicious
        if self.ui.radioButton.isChecked():
            location = "Tarantula" + ".csv"
            suspicious = TarantulaError(spectrum)

        if self.ui.radioButton_2.isChecked():
            location = "Ochiai" + ".csv"

            suspicious = OchiaiError(spectrum)

        if self.ui.radioButton_3.isChecked():
            location = "D_" + ".csv"
            suspicious = D_star(spectrum)
        self.error_location = location

        # QTreeWidgetItem(self.root).setText(0, self.error_location)

        return suspicious

    # Order of suspiciousness
    def errorLocated(self):
        trainableLayers = get_trainable_layers(self.model)
        neuron_suspicious = []
        for layer in trainableLayers:
            file_name = "layer" + str(layer) + "-" + ".txt"
            path = self.spectrum_path + "/" + file_name
            # path = self.spectrum_path + "/layer" + str(layer) + ".txt"
            spectrum = np.loadtxt(path)
            number_neuron = len(spectrum)
            suspicious = self.getSus(spectrum)
            suspicious = suspicious.reshape((number_neuron, 1))  # Degree of suspicion
            layer_index = np.ones((number_neuron, 1)) * int(layer)  # layer
            neuron_index = np.arange(0, len(spectrum)).reshape((number_neuron, 1))  # neuron

            # Merging layer, neuron, suspicious degree
            temp = np.concatenate((layer_index, neuron_index), axis=1)
            error_Location = np.concatenate((temp, suspicious), axis=1)
            neuron_suspicious.append(error_Location)

        suspicious = neuron_suspicious[0]
        for i in range(1, len(neuron_suspicious)):
            suspicious = np.concatenate([suspicious, neuron_suspicious[i]])
            print(suspicious.shape)
        error_Location = suspicious[argsort(-suspicious[:, 2])]
        print(error_Location)
        file = self.project_path + "/" + self.error_location
        np.savetxt(file + ".txt", error_Location)

        self.location_path = file + ".txt"

        generateErrorCsv(file, error_Location)
        # if self.inChild(self.root, error_Location):
        #    QTreeWidgetItem(self.root).setText(0, self.error_location)
        QTreeWidgetItem(self.root).setText(0, self.error_location)
        self.ui.logtextBrowser.append("Suspicious degree sorting succeeded......")
        self.ui.logtextBrowser.append("*" * 50)

    # Parameter Configuration Path
    def getParSrc(self):
        self.parsrc = QFileDialog.getExistingDirectory()
        self.ui.browseLineEdit.setText(self.parsrc)

    # Optimizing the acquisition path
    def getOpenSrc(self):
        self.opensrc = QFileDialog.getExistingDirectory()
        self.ui.optimizaLineEdit.setText(self.opensrc)

    # Neuronal spectrum preservation path
    def getsSpectrumPath(self):
        # self.spectrum_path = QFileDialog.getExistingDirectory()
        self.spectrum_path = self.project_path
        self.ui.lineEdit_7.setText(self.project_path)

    # Generated neuron spectrum
    def generate_spectrum(self):
        self.test = generate_data(self.test_path)
        print(self.test.shape)
        if self.label_path == None:
            self.label = self.model.predict(self.test / 255)
        else:
            self.label = generate_label(self.label_path)
        if self.meta_path is not None:
            self.test = generate_data(self.meta_path)
        print(self.test.shape)
        self.pre_label = self.model.predict(self.test / 255)
        print(self.pre_label)
        trainableLayers = get_trainable_layers(self.model)

        for l in trainableLayers:
            print(l)
            file_name = "layer" + str(l) + "-" + ".txt"
            path = self.spectrum_path + "/" + file_name
            sub_model = tf.keras.Model(inputs=self.model.input, outputs=self.model.get_layer(index=l).output)
            output = sub_model.predict(self.test)
            generateSpectrum(output, path, self.label, self.pre_label, 2 / 25)
            if self.inChild(self.root, file_name):
                QTreeWidgetItem(self.root).setText(0, file_name)

        self.ui.logtextBrowser.append("Spectrum information has been saved to" + self.spectrum_path)
        self.ui.logtextBrowser.append("*" * 50)

    # Generate test cases for the neuronal spectrum
    def generate_spectrum1(self):
        self.generate_testcase_label_path = new_input + "/steering1.csv"
        self.test = generateData_jpg(self.generate_save_path) / 255.0
        self.label = generate_testcase_label1(self.generate_testcase_label_path)
        self.pre_label = generate_testcase_label2(self.generate_testcase_label_path)

        trainableLayers = get_trainable_layers(self.model)

        for l in trainableLayers:
            print(l)
            path = self.spectrum_path + "/layer" + str(l) + ".txt"
            sub_model = tf.keras.Model(inputs=self.model.input, outputs=self.model.get_layer(index=l).output)
            output = sub_model.predict(self.test)
            generateSpectrum(output, path, self.label, self.pre_label, 2 / 25)
        self.ui.logtextBrowser.append("Spectrum information has been saved to" + self.spectrum_path)
        self.ui.logtextBrowser.append("*" * 50)

    # Generate a transformation image
    def generateMeta(self):
        self.ui.logtextBrowser.append("Generating a metamorphosis image......")
        datasetPath = self.test_path
        metaPath = self.meta_path
        fileList = []
        # The image path is stored to fileList
        for file in sorted(os.listdir(datasetPath)):
            if file.endswith(".jpg") or file.endswith(".jpg"):
                fileList.append(file)
                print(file)
        random.seed(504)
        trans_list = []
        if self.ui.checkBox_4.isChecked() == True:
            trans_list.append(7)
        if self.ui.checkBox_2.isChecked() == True:
            trans_list.append(1)
            trans_list.append(2)
            trans_list.append(3)
            trans_list.append(4)
        if self.ui.checkBox.isChecked() == True:
            trans_list.append(5)
            trans_list.append(6)

        for i in range(len(fileList)):
            index_function = random.choice(trans_list)
            index = random.randint(1, 10)
            seed_image = cv2.imread(os.path.join(datasetPath, fileList[i]))
            metaImag = precessImag(index_function, index, seed_image)
            cv2.imwrite(os.path.join(metaPath, str(fileList[i])), metaImag)
        # QTreeWidgetItem(self.root).setText(0, self.meta_path)
        self.ui.logtextBrowser.append("The metamorphosed image is saved in" + self.meta_path)
        self.ui.logtextBrowser.append("*" * 50)

    # Transformation image save path
    # def getMetaPath(self):
    #     self.meta_path = self.meta_path_temp
    #     self.ui.lineEdit_6.setText(self.meta_path)

    # Setting Coverage
    def saveCover_setting(self):
        self.ui.logtextBrowser.append("The coverage was successfully selected. Procedure")
        self.ui.logtextBrowser.append("*" * 50)

        # self.ui.coverage_lineEdit.setText(coverage_name)

    # Seed image coverage
    def seedcoverage(self):
        model = self.model
        seed_cover_path = self.project_path + "/seed_coverage.csv"
        if self.ui.self_defining.isChecked() == True:
            # self.test = generate_data(self.test_path)
            # self.test /= 255
            self.seed_coverage, coverage = self.calculate_coverage(model, self.test_path, seed_cover_path)
        elif self.ui.Minist_radioButton.isChecked() == True:
            self.test_path = self.load_minist_data()
            self.seed_coverage, coverage = self.calculate_coverage(model, self.test_path, seed_cover_path)
        elif self.ui.Ciafr_radioButton.isChecked() == True:
            self.test_path = self.load_cifar10_data()
            self.seed_coverage, coverage = self.calculate_coverage(model, self.test_path, seed_cover_path)
        elif self.ui.FashionMinist_radioButton.isChecked() == True:
            self.test_path = self.load_fashion_mnist_data()
            self.seed_coverage, coverage = self.calculate_coverage(model, self.test_path, seed_cover_path)
        if self.inChild(self.root, "seed_coverage.csv"):
            QTreeWidgetItem(self.root).setText(0, "seed_coverage.csv")
        self.ui.logtextBrowser.append("The current seed data set coverage is：{}".format(coverage))
        self.ui.logtextBrowser.append("*" * 50)
        coverage_name = ""
        for key in self.seed_coverage.keys():
            coverage_name = coverage_name + key + "\n"
        self.ui.coverage_lineEdit.setText(coverage_name)

    # Seed text coverage
    def seed_text_coverage(self):
        model = self.model
        seed_text_path = self.test_path
        seed_cover_path = self.project_path + "/seed_coverage.csv"
        all_data = self.read_generate(seed_text_path)

        all_data = all_data.copy()
        all_data = np.array(all_data)

        self.seed_coverage, coverage = self.calculate_text_coverage(model, all_data, seed_cover_path)

        if self.inChild(self.root, "seed_coverage.csv"):
            QTreeWidgetItem(self.root).setText(0, "seed_coverage.csv")
        self.ui.logtextBrowser.append("The current seed data set coverage is：{}".format(coverage))
        self.ui.logtextBrowser.append("*" * 50)
        coverage_name = ""
        for key in self.seed_coverage.keys():
            coverage_name = coverage_name + key + "\n"
        self.ui.coverage_lineEdit.setText(coverage_name)

    # Gets the file path for the upper and lower boundaries
    def generateMin_Max(self):
        min_max_file_name = "min_max.npy"
        model = self.model

        if self.ui.self_defining.isChecked() == True:
            data = generateData_jpg(self.test_path)
        elif self.ui.Minist_radioButton.isChecked() == True:
            data = self.load_minist_data()
        elif self.ui.Ciafr_radioButton.isChecked() == True:
            data = self.load_cifar10_data()

        elif self.ui.FashionMinist_radioButton.isChecked() == True:
            data = self.load_fashion_mnist_data()

        self.min_max_file = self.project_path + "/" + min_max_file_name
        getMin_Max(model, data, self.min_max_file)
        self.ui.logtextBrowser.append("The upper and lower boundaries of the neuron were successfully generated...")
        self.ui.logtextBrowser.append("*" * 50)
        if self.inChild(self.root, min_max_file_name):
            QTreeWidgetItem(self.root).setText(0, min_max_file_name)

    # Create text minmax
    def generateMin_Max_tf(self):
        min_max_file_name = "min_max.npy"
        model = self.model

        data = pd.read_csv(self.test_path, header=None)
        data = np.array(data)

        self.min_max_file = self.project_path + "/" + min_max_file_name
        getMin_Max(model, data, self.min_max_file)
        self.ui.logtextBrowser.append("The upper and lower boundaries of the neuron were successfully generated...")
        self.ui.logtextBrowser.append("*" * 50)
        if self.inChild(self.root, min_max_file_name):
            QTreeWidgetItem(self.root).setText(0, min_max_file_name)

    # Gets the file path for the upper and lower boundaries
    def getMin_MaxPath(self):
        # selected_file = QFileDialog.getOpenFileName()
        # self.min_max_file = selected_file[0]
        self.min_max_file = self.project_path + "/min_max.npy"
        print(self.min_max_file)
        self.ui.lineEdit_12.setText(self.min_max_file)

    # Calculated coverage rate
    def calculate_coverage(self, model, testcase_path, cover_path):
        # Here is the debugging code
        # print(model)
        # print(testcase_path)
        # print(cover_path)
        # exit(0)
        # End of debugging code

        print("---------------------------------")
        # print("For all test cases shape:", testcase_path.shape)
        # Debug code
        # self.test = tf.convert_to_tensor(self.test)

        # Processing Pictures
        print(testcase_path)
        self.test = generate_data(testcase_path)
        self.test /= 255
        # Get the output
        output = getOutPut(model, self.test)

        k1 = int(self.ui.comboBox.currentText())
        k2 = int(self.ui.comboBox_2.currentText())

        self.coverDic = {}
        # Coverage of neurons ac
        if self.ui.cover_1.isChecked() == True:
            nc, activate = neuronCover(output)
            self.neuron_Cover.append(nc)
            self.neuron_Cover.append(activate)
            self.coverDic["神经元覆盖率"] = nc

        # K- Multinode neuron coverage
        if self.ui.cover_2.isChecked() == True:
            knc = KMNCov(output, k1, self.min_max_file)
            self.KMN_Cov = knc
            self.coverDic["K-多节神经元覆盖率"] = knc

        # Neuronal boundary coverage
        if self.ui.cover_3.isChecked() == True or self.ui.cover_4.isChecked() == True:
            nbc, Upper = NBCov(output, self.min_max_file)
            self.NB_Cov = nbc
            self.coverDic["神经元边界覆盖率"] = nbc

        # Strong neuronal activation coverage
        if self.ui.cover_4.isChecked() == True:
            snc = SNACov(output, Upper)
            self.SNA_Cov = snc
            self.coverDic["强神经元激活覆盖率"] = snc

        # top-k Coverage of neurons
        if self.ui.cover_5.isChecked() == True:
            tknc = TKNCov(self.model, self.test, k2)
            self.TKN_Cov = tknc
            self.coverDic["top-k神经元覆盖率"] = tknc

        # print("神经元覆盖率:{}%\nK-多节神经元覆盖率:{}%\n神经元边界覆盖率:{}%\n"
        #       "强神经元激活覆盖率:{}%\ntop-k神经元覆盖率:{}%".format(nc * 100, knc * 100, nbc * 100, snc * 100,
        #                                               tknc * 100))
        coverage = ""
        for key in self.coverDic.keys():
            value = round(self.coverDic[key] * 100, 2)
            coverage = coverage + key + ":" + str(value) + "%" + "   "

        if os.path.exists(cover_path) == False:
            create_csv(cover_path, self.coverDic)
        else:
            append_csv(cover_path, self.coverDic)

        return self.coverDic, coverage

    # The chart shows the coverage of the seed image
    def show_seed_coverage(self):
        mpl.rcParams['font.sans-serif'] = ['KaiTi', 'SimHei', 'FangSong']  # Chinese font, the preferred use of regular, if you can not find the regular, then use black
        mpl.rcParams['font.size'] = 12  # Font size
        mpl.rcParams['axes.unicode_minus'] = False  # The negative sign is displayed normally

        x = list(self.coverDic.keys())
        y = list(self.coverDic.values())

        index = [i for i in range(len(x))]

        x1 = np.arange(len(x))
        bar_width = 0.2

        plt.bar(x1, y, bar_width, align="center", color="c", label="种子图片", alpha=0.5)

        ax = plt.gca()
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')

        plt.xticks(x1, x, rotation=45, fontsize=9)
        plt.legend()

        plt.xlabel("覆盖率类型", fontdict={'size': 16})
        plt.ylabel("覆盖率", fontdict={'size': 16})
        plt.title(u'种子图片覆盖率', fontdict={'size': 16})

        for a, b in zip(index, y):
            plt.text(a, b + 0.01, '%.3f' % b, ha='center', va='bottom', fontsize=9)

        plt.savefig(self.cover_path + "/seed coverage.jpg", dpi=200, bbox_inches='tight')
        plt.show()

    # Calculate the coverage of each image generated
    def getImageCover(self):
        self.generate_cov_csv = self.generate_save_path + '/' + "steering1.csv"
        with open(self.generate_cov_csv, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['序号', '种子图片', '生成图片', '原标签', '模型预测标签', '覆蓋率'])

        datasetPath = self.generate_save_path
        fileList = []
        # The image path is stored to fileList
        for file in sorted(os.listdir(datasetPath)):
            if file.endswith(".jpg"):
                fileList.append(file)
        # fileList = getSmallData(fileList)  # Get 500 data
        for i in range(len(fileList)):
            path = os.path.join(datasetPath, fileList[i])
            self.test = preprocess_image(path)  # Pre-processed image

            self.pre_label = self.model.predict(self.test / 255)  # Model prediction results
            model_predict_label = self.pre_label[0][0]

            self.test_name = fileList[i]
            self.neuron_Cover = []  # self.neuron_Cover[0] Neuron coverage self.neuron_Cover[1] Activation matrix

            self.test /= 255
            output = getOutPut(self.model, self.test)

            k1 = int(self.ui.comboBox.currentText())
            k2 = int(self.ui.comboBox_2.currentText())

            coverDic1 = {}
            # Neuron coverage ac
            if self.ui.cover_1.isChecked() == True:
                nc, activate = neuronCover(output)
                self.neuron_Cover.append(nc)
                self.neuron_Cover.append(activate)
                coverDic1["神经元覆盖率"] = nc

            # K- multinode neuron coverage
            if self.ui.cover_2.isChecked() == True:
                knc = KMNCov(output, k1, self.min_max_file)
                self.KMN_Cov = knc
                coverDic1["K-多节神经元覆盖率"] = knc

            # Neuronal boundary coverage
            if self.ui.cover_3.isChecked() == True or self.ui.cover_4.isChecked() == True:
                nbc, Upper = NBCov(output, self.min_max_file)
                self.NB_Cov = nbc
                coverDic1["神经元边界覆盖率"] = nbc

            # Strong neuronal activation coverage
            if self.ui.cover_4.isChecked() == True:
                snc = SNACov(output, Upper)
                self.SNA_Cov = snc
                coverDic1["强神经元激活覆盖率"] = snc

            # top-k Coverage of neurons
            if self.ui.cover_5.isChecked() == True:
                tknc = TKNCov(self.model, self.test, k2)
                self.TKN_Cov = tknc
                coverDic1["top-k神经元覆盖率"] = tknc

            print("神经元覆盖率:{}%\nK-多节神经元覆盖率:{}%\n神经元边界覆盖率:{}%\n"
                  "强神经元激活覆盖率:{}%\ntop-k神经元覆盖率:{}%".format(nc * 100, knc * 100, nbc * 100, snc * 100,
                                                          tknc * 100))
            coverage = ""
            for key in coverDic1.keys():
                value = round(coverDic1[key] * 100, 2)
                coverage = coverage + key + ":" + str(value) + "%" + "   "

            self.generate_csv = self.generate_save_path + '/' + "steering.csv"
            with open(self.generate_csv, 'r') as csvfile:
                reader = csv.reader(csvfile)
                head = next(reader)
                for row in reader:
                    id = row[0]
                    seed_image = row[1]
                    generate_image = row[2]
                    label = row[3]
                    if generate_image == self.test_name:
                        row.append(model_predict_label)
                        row.append(coverage)
                        with open(self.generate_cov_csv, 'a', newline='') as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow(row)

    # display
    def show(self):
        self.ui.show()

    # Display the generated results
    def OpenResult(self):
        # self.getImageCover()
        self.result = ResultTest()
        self.result.show()

    # Obtaining the resource location
    # def getSeedsrc(self):
    #     # self.seed_path = QFileDialog.getExistingDirectory()
    #     # self.ui.seedLineEdit_2.setText(self.seed_path)
    #     self.ui.logtextBrowser.append("成功导入种子数据集：" + self.seed_path)
    #     self.ui.logtextBrowser.append("*"*50)
    #
    # def getSeedLabelsrc(self):
    #     # seed_label_path = QFileDialog.getOpenFileName()
    #     # self.label_path = seed_label_path[0]
    #     # self.ui.seedlabel_LineEdit_2.setText(self.label_path)
    #     self.ui.logtextBrowser.append("成功导入种子数据集标签：" + self.label_path)
    #     self.ui.logtextBrowser.append("*" * 50)

    # Select the path to save the generated test case -- image data
    def getSaveSrc(self):
        # self.generate_save_path = QFileDialog.getExistingDirectory()
        self.ui.genlocalLineEdit_2.setText(self.generate_save_path)
        self.ui.logtextBrowser.append("成功选择生成测试用例保存路径：" + self.generate_save_path)
        self.ui.logtextBrowser.append("*" * 50)

    # Select the path to generate test cases -- text data
    def get_text_SaveSrc(self):
        # self.generate_save_path = QFileDialog.getExistingDirectory()
        self.ui.textLineEdit.setText(self.generate_save_path)
        self.ui.logtextBrowser.append("成功选择生成测试用例保存路径：" + self.generate_save_path)
        self.ui.logtextBrowser.append("*" * 50)

    # The chart shows
    def show_figure(self, coverDic, coverage, figure_save_path):
        mpl.rcParams['font.sans-serif'] = ['KaiTi', 'SimHei', 'FangSong']  # Chinese font, the preferred use of regular, if you can not find the regular, then use black
        mpl.rcParams['font.size'] = 12  # Font size
        mpl.rcParams['axes.unicode_minus'] = False  # The negative sign is displayed normally

        x = list(coverage.keys())
        y1 = list(coverDic.values())
        y2 = list(coverage.values())

        index = [i for i in range(len(x))]

        x1 = np.arange(len(x))
        bar_width = 0.2

        plt.bar(x1, y1, bar_width, align="center", color="c", label="种子图片", alpha=0.5)
        plt.bar(x1 + bar_width, y2, bar_width, align="center", color="b", label="生成图片", alpha=0.5)

        ax = plt.gca()
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')

        plt.xticks(x1 + bar_width / 2, x, rotation=45, fontsize=9)
        plt.legend()

        plt.xlabel("覆盖率类型", fontdict={'size': 16})
        plt.ylabel("覆盖率", fontdict={'size': 16})
        plt.title(u'覆盖率比较', fontdict={'size': 16})

        for a, b in zip(index, y1):
            plt.text(a, b + 0.01, '%.3f' % b, ha='center', va='bottom', fontsize=8)
        for a, b in zip(index, y2):
            plt.text(a + bar_width, b + 0.01, '%.3f' % b, ha='center', va='bottom', fontsize=8)

        plt.savefig(figure_save_path, dpi=200, bbox_inches='tight')
        plt.show()

    # Text data generation
    def generate_text_data(self):
        generate_number = self.ui.number_spinBox.value()
        dimension = self.ui.dimension_spinBox.value()
        lower = self.ui.lower_spinBox.value()
        upper = self.ui.upper_spinBox.value()
        generate_text_path = self.generate_save_path + "/generate_text_data.csv"

        generate_text_testcase(generate_number, generate_text_path, dimension, lower, upper)
        self.generate_number = str(generate_number)
        self.ui.testcase_lineEdit.setText(self.generate_number)
        QTreeWidgetItem(self.root).setText(0, generate_text_path)
        self.ui.logtextBrowser.append("成功生成测试用例...")
        self.ui.logtextBrowser.append("*" * 50)

    # Image data generation
    def getGenSet(self):
        global new_input
        # dataset_path = self.seed_path
        dataset_path = self.test_path
        seed_label_path = self.label_path
        new_input = self.generate_save_path
        startticks = time.time()
        maxtime = self.ui.timeSpinBox_2.value()
        maxgeneratenumber = self.ui.numSpinBox_2.value()
        maxchangenumber = self.ui.changeSpinBox_2.value()

        #  Transformation parameter acquisition
        py_1 = self.ui.py_spinBox_1.value()
        py_2 = self.ui.py_spinBox_2.value()
        sf_1 = self.ui.sf_spinBox_1.value()
        sf_2 = self.ui.sf_spinBox_2.value()
        jq_1 = self.ui.jq_spinBox_1.value()
        jq_2 = self.ui.jq_spinBox_2.value()
        xz_1 = self.ui.xz_spinBox_1.value()
        xz_2 = self.ui.xz_spinBox_2.value()
        db_1 = self.ui.db_spinBox_1.value()
        db_2 = self.ui.db_spinBox_2.value()
        ld_1 = self.ui.ld_spinBox_1.value()
        ld_2 = self.ui.ld_spinBox_2.value()
        mh_1 = self.ui.mh_spinBox_1.value()
        mh_2 = self.ui.mh_spinBox_2.value()

        if self.ui.gen_radioButton.isChecked() == True:

            generater = rambo_guided1(dataset_path, seed_label_path, new_input, startticks, maxtime, maxgeneratenumber,
                                      maxchangenumber, py_1, py_2, sf_1, sf_2, jq_1, jq_2, xz_1, xz_2, db_1, db_2, ld_1,
                                      ld_2, mh_1, mh_2)
            # generater = rambo_guided(dataset_path, seed_label_path, new_input, startticks, maxtime,
            #              maxgeneratenumber, maxchangenumber)

        elif self.ui.cover_radioButton.isChecked() == True:

            cover_1 = self.ui.cover_1.isChecked()
            cover_2 = self.ui.cover_2.isChecked()
            cover_3 = self.ui.cover_3.isChecked()
            cover_4 = self.ui.cover_4.isChecked()
            cover_5 = self.ui.cover_5.isChecked()

            nc = self.ui.NC_comboBox.currentText().split("%")[0]
            knc = self.ui.KMNC_comboBox.currentText().split("%")[0]
            nbc = self.ui.NBC_comboBox.currentText().split("%")[0]
            snc = self.ui.SNAC_comboBox.currentText().split("%")[0]
            tknc = self.ui.TKNC_comboBox.currentText().split("%")[0]

            k1 = int(self.ui.comboBox.currentText())
            k2 = int(self.ui.comboBox_2.currentText())

            print(nc, knc, nbc, snc, tknc)
            print(k1, k2)

            generater = rambo_guided2(dataset_path, seed_label_path, new_input, self.min_max_file, maxchangenumber,
                                      py_1, py_2, sf_1, sf_2, jq_1, jq_2, xz_1, xz_2, db_1, db_2, ld_1, ld_2, mh_1,
                                      mh_2, k1, k2, self.model, cover_1, cover_2, cover_3, cover_4, cover_5, nc, knc,
                                      nbc, snc, tknc)
            # generater = rambo_guided(dataset_path, seed_label_path, new_input, startticks, maxtime,
            #              maxgeneratenumber, maxchangenumber)

        self.generatenumber = str(generater)
        self.ui.number_label.setText(self.generatenumber)
        self.ui.testcase_lineEdit.setText(self.generatenumber)
        QTreeWidgetItem(self.root).setText(0, new_input)
        self.ui.logtextBrowser.append("生成测试用例中...")
        self.ui.logtextBrowser.append("*" * 50)

    # Generate coverage -- image data
    def generate_coverage(self):
        model = self.model
        generate_coverage_path = self.generate_save_path + "/generate_coverage.csv"
        self.show_generate_coverage = ""
        # self.generate_save_path = generate_data(self.generate_save_path)
        # self.generate_save_path /= 255
        self.generate_cover, coverage = self.calculate_coverage(model, self.generate_save_path,
                                                                   generate_coverage_path)
        for generate_key in self.generate_cover.keys():
            generate_value = round(self.generate_cover[generate_key] * 100, 2)
            self.show_generate_coverage = self.show_generate_coverage + generate_key + ":" + str(
                generate_value) + "%" + "\n"
        self.ui.coverage_label.setText(self.show_generate_coverage)

        # self.getImageCover()

        self.ui.cov_lineEdit.setText(self.show_generate_coverage)

    # Generate coverage -- text
    def calculate_text_coverage(self, model, all_data, cover_path):

        k1 = int(self.ui.comboBox.currentText())
        k2 = int(self.ui.comboBox_2.currentText())

        self.coverDic = {}
        output = getOutPut(model, all_data)
        # output = getOutPut(model, x_test)
        print(output.shape)
        if self.ui.cover_1.isChecked() == True:
            nc, activate = neuronCover(output)
            self.neuron_Cover.append(nc)
            self.neuron_Cover.append(activate)
            self.coverDic["神经元覆盖率"] = nc

        # K- multinode neuron coverage
        if self.ui.cover_2.isChecked() == True:
            knc = KMNCov(output, k1, self.min_max_file)
            self.KMN_Cov = knc
            self.coverDic["K-多节神经元覆盖率"] = knc

        # Neuronal boundary coverage
        if self.ui.cover_3.isChecked() == True or self.ui.cover_4.isChecked() == True:
            nbc, Upper = NBCov(output, self.min_max_file)
            self.NB_Cov = nbc
            self.coverDic["神经元边界覆盖率"] = nbc

        # Strong neuronal activation coverage
        if self.ui.cover_4.isChecked() == True:
            snc = SNACov(output, Upper)
            self.SNA_Cov = snc
            self.coverDic["强神经元激活覆盖率"] = snc

        # top-k Coverage of neurons
        if self.ui.cover_5.isChecked() == True:
            tknc = TKNCov(self.model, all_data, k2)
            self.TKN_Cov = tknc
            self.coverDic["top-k神经元覆盖率"] = tknc

        # print("神经元覆盖率:{}%\nK-多节神经元覆盖率:{}%\n神经元边界覆盖率:{}%\n"
        #       "强神经元激活覆盖率:{}%\ntop-k神经元覆盖率:{}%".format(nc * 100, knc * 100, nbc * 100, snc * 100,
        #                                               tknc * 100))
        coverage = ""
        for key in self.coverDic.keys():
            value = round(self.coverDic[key] * 100, 2)
            coverage = coverage + key + ":" + str(value) + "%" + "   "

        if os.path.exists(cover_path) == False:
            create_csv(cover_path, self.coverDic)
        else:
            append_csv(cover_path, self.coverDic)

        return self.coverDic, coverage

    def read_generate(self, generate_name):
        all_data = list()
        with open(generate_name, "r", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                data = list()
                for i in row:
                    i = float(i)
                    data.append(i)
                all_data.append(data)
        return all_data

    # Generate coverage -- textual data
    def generate_text_coverage(self):
        model = self.model
        generate_text_path = self.generate_save_path + "/generate_text_data.csv"
        generate_coverage_path = self.generate_save_path + "/generate_text_coverage.csv"
        self.show_generate_coverage = ""

        all_data = self.read_generate(generate_text_path)

        all_data = all_data.copy()
        all_data = np.array(all_data)

        self.generate_cover, coverage = self.calculate_text_coverage(model, all_data,
                                                                        generate_coverage_path)
        for generate_key in self.generate_cover.keys():
            generate_value = round(self.generate_cover[generate_key] * 100, 2)
            self.show_generate_coverage = self.show_generate_coverage + generate_key + ":" + str(
                generate_value) + "%" + "\n"
        self.ui.text_cov_label.setText(self.show_generate_coverage)

        # self.getImageCover()

        self.ui.cov_lineEdit.setText(self.show_generate_coverage)

    # Generate coverage comparison -- image data
    def generate_notext_coverage_compare(self):
        figure_save_path = new_input + "/generate_coverage_compare.jpg"
        self.show_figure(self.seed_coverage, self.generate_cover, figure_save_path)
        self.ui.logtextBrowser.append("成功比较测试用例的覆盖率...")
        self.ui.logtextBrowser.append("*" * 50)

    # Generate coverage comparisons -- textual data
    def generate_text_coverage_compare(self):
        figure_save_path = self.generate_save_path + "/generate_coverage_compare.jpg"
        self.show_figure(self.seed_coverage, self.generate_cover, figure_save_path)
        # QTreeWidgetItem(self.root).setText(0, figure_save_path)
        self.ui.logtextBrowser.append("成功比较测试用例的覆盖率...")
        self.ui.logtextBrowser.append("*" * 50)

    def show(self):
        self.ui.show()

    # Choosing an optimization path
    def getOptimizePath(self):
        # self.optimize_path = QFileDialog.getExistingDirectory()
        self.ui.optimizaLineEdit.setText(self.optimize_path)
        self.ui.logtextBrowser.append("成功选择优化测试用例保存路径：" + self.optimize_path)
        self.ui.logtextBrowser.append("*" * 50)

    # Optimize the picture
    def openOptimize(self):
        model_path = self.model_path  # Path of model
        seed_path = self.test_path  # Seed picture path
        seed_label_path = self.label_path  # The tag path corresponding to the seed image
        generate_path = self.test_path  # The path to the generated image
        genenrate_label_path = self.label_path  # The path to the tag corresponding to the generated image
        optimize_output_path = self.optimize_path  # Save path of the optimized image
        if self.ui.text_radioButton.isChecked() == True:
            generate_text_path = self.generate_save_path + "/generate_text_data.csv"  # The path to the generated text
            result_flag = optimization(model_path, self.min_max_file, generate_text_path,
                                       optimize_output_path + "/optimize_text_data.csv")
            QTreeWidgetItem(self.root).setText(0, optimize_output_path + "/optimize_text_data.csv")
            self.ui.logtextBrowser.append("测试用例优化中...")
            self.ui.logtextBrowser.append("*" * 50)
            if result_flag == True:
                self.ui.logtextBrowser.append("测试用例优化成功...")
            else:
                self.ui.logtextBrowser.append("测试用例优化失败...")
            self.ui.logtextBrowser.append("*" * 50)
        elif self.ui.concolicButton.isChecked() == True:
            concolic_main(model_path, seed_path, seed_label_path, optimize_output_path)  # Call a function
            QTreeWidgetItem(self.root).setText(0, optimize_output_path)
            self.ui.logtextBrowser.append("测试用例优化中...")
            self.ui.logtextBrowser.append("*" * 50)
        elif self.ui.rbluradioButton.isChecked() == True:
            run_fuzzer(model_path, seed_path, seed_label_path, optimize_output_path)
            QTreeWidgetItem(self.root).setText(0, optimize_output_path)
            self.ui.logtextBrowser.append("测试用例优化中...")
            self.ui.logtextBrowser.append("*" * 50)

    # Optimize image coverage
    def OptimizeCoverage(self):
        model = self.model
        optimize_coverage_path = self.optimize_path + "/optimize_coverage.csv"
        print(optimize_coverage_path)
        # exit(0)
        self.show_optimize_coverage = ""

        if self.ui.text_radioButton.isChecked() == True:
            all_data = self.read_generate(self.optimize_path + "/optimize_text_data.csv")
            all_data = all_data.copy()
            all_data = np.array(all_data)
            self.optimize_coverage, coverage = self.calculate_text_coverage(model, all_data, optimize_coverage_path)
        else:
            self.optimize_coverage, coverage = self.calculate_coverage(model, self.optimize_path,
                                                                       optimize_coverage_path)
        for optimize_key in self.optimize_coverage.keys():
            optimize_value = round(self.optimize_coverage[optimize_key] * 100, 2)
            self.show_optimize_coverage = self.show_optimize_coverage + optimize_key + ":" + str(
                optimize_value) + "%" + "\n"

        self.ui.logtextBrowser.setText(self.show_optimize_coverage)

        self.ui.logtextBrowser.append("成功计算优化测试用例的覆盖率...")
        self.ui.logtextBrowser.append("*" * 50)

    # Optimize image coverage comparison
    def OptimizeCoverageCompare(self):
        figure_save_path = self.optimize_path + "/optimize coverage compare.jpg"
        self.show_figure(self.generate_cover, self.optimize_coverage, figure_save_path)
        QTreeWidgetItem(self.root).setText(0, figure_save_path)
        self.ui.logtextBrowser.append("成功比较优化前后测试用例的覆盖率...")
        self.ui.logtextBrowser.append("*" * 50)

    # Robustness to power
    # 1. Importing the original model
    def import_model_robustness(self):
        selected_file = QFileDialog.getOpenFileName()
        self.model_robustness_path = selected_file[0]
        self.ui.model_path.setText(self.model_robustness_path)
        self.ui.logtextBrowser.append("成功选择原始模型：" + self.model_robustness_path)
        self.ui.logtextBrowser.append("*" * 50)
        self.model_robustness = tf.keras.models.load_model(self.model_robustness_path)

    # 2. Import the repaired model
    def import_repaired_model_robustness(self):
        selected_file = QFileDialog.getOpenFileName()
        self.repaired_model_robustness_path = selected_file[0]
        self.ui.repaired_model_path.setText(self.repaired_model_robustness_path)
        self.ui.logtextBrowser.append("成功选择修复后的模型：" + self.repaired_model_robustness_path)
        self.ui.logtextBrowser.append("*" * 50)
        self.repaired_model_robustness = tf.keras.models.load_model(self.repaired_model_robustness_path)

    # 3. Import the original data set
    def import_dataset_robustness(self):
        self.dataset_robustness_path = QFileDialog.getExistingDirectory()
        self.ui.dataset_path.setText(self.dataset_robustness_path)
        self.ui.logtextBrowser.append("成功导入数据集：" + self.dataset_robustness_path)
        self.ui.logtextBrowser.append("*" * 50)

    # 4. Import the raw data set label
    def import_dataset_label_robustness(self):
        selected_file = QFileDialog.getOpenFileName(self.ui, "选择数据集标签", ".", "CSV(逗号分隔)(*.csv)")
        self.dataset_label_robustness_path = selected_file[0]
        self.ui.dataset_label_path.setText(self.dataset_label_robustness_path)
        self.ui.logtextBrowser.append("成功导入数据集：" + self.dataset_label_robustness_path)
        self.ui.logtextBrowser.append("*" * 50)

    # 4. Select the path to save the counter sample
    def choose_countermeasure_save_path(self):
        self.countermeasure_save_path = self.robustness_path
        self.ui.countermeasure_robustness.setText(self.countermeasure_save_path)
        self.ui.logtextBrowser.append("成功选择对抗样本保存路径：" + self.countermeasure_save_path)
        self.ui.logtextBrowser.append("*" * 50)

    # 5. Generate Antagonistic samples
    def generate_countermeasure(self):
        model = self.model_robustness
        repaired_model = self.repaired_model_robustness
        dataset_path = self.dataset_robustness_path
        countermeasure_save_path = self.robustness_path
        save_model_predict_countermeasure_label_path = self.robustness_path + "\model predict countermeasure label.csv"
        save_repaired_model_predict_countermeasure_label_path = self.robustness_path + r"\repaired model predict countermeasure label.csv"

        generate_countermeasure_samples(model, repaired_model, dataset_path, countermeasure_save_path,
                                        save_model_predict_countermeasure_label_path,
                                        save_repaired_model_predict_countermeasure_label_path)
        self.ui.logtextBrowser.append("成功生成对抗样本......")
        self.ui.logtextBrowser.append("*" * 50)

    # 6. Calculated coverage rate
    def calculate_robustness(self):
        model = self.model_robustness
        repaired_model = self.repaired_model_robustness
        dataset_path = self.dataset_robustness_path
        countermeasure_save_path = self.robustness_path
        dataset_label_path = self.dataset_label_robustness_path
        save_model_predict_dataset_label_path = self.robustness_path + "\model predict dataset label.csv"
        save_repaired_model_predict_dataset_label_path = self.robustness_path + r"\repaired model predict dataset label.csv"
        save_model_predict_countermeasure_label_path = self.robustness_path + "\model predict countermeasure label.csv"
        save_repaired_model_predict_countermeasure_label_path = self.robustness_path + r"\repaired model predict countermeasure label.csv"

        # List of images from the original data set
        dataset_list = generate_data_robustness(dataset_path)
        # List of original dataset tags (list of correct tags) L_correct
        dataset_label_list = generate_label_robustness(dataset_label_path)

        # The original model predicts the original data set label
        predict_dataset_label(model, dataset_path, save_model_predict_dataset_label_path)
        # The original model predicts the original data set tag list L_OEOM
        model_predict_label_list = generate_label_robustness(save_model_predict_dataset_label_path)

        # Against sample picture list
        countermeasure_list = generate_data_robustness(countermeasure_save_path)

        # List of anti-sample labels predicted by the original model L_AEOM
        model_predict_countermeasure_label_path = generate_label_robustness(
            save_model_predict_countermeasure_label_path)

        # The original dataset label predicted by the repaired model
        predict_dataset_label(repaired_model, dataset_path, save_repaired_model_predict_dataset_label_path)
        # List of original data set tags predicted by the repaired model  L_OERM
        repaired_model_predict_dataset_label_list = generate_label_robustness(
            save_repaired_model_predict_dataset_label_path)

        # Tag list of the adversarial sample dataset predicted by the repaired model L_AERM
        repaired_model_predict_countermeasure_label_list = generate_label_robustness(
            save_repaired_model_predict_countermeasure_label_path)

        # Whether the original model predicted against the sample picture should be judged correctly
        L_Describe_1 = describe(save_model_predict_countermeasure_label_path)

        # Whether the prediction of the model against the sample picture after repair should be judged correctly
        L_Describe_2 = describe(save_repaired_model_predict_countermeasure_label_path)

        robustness = judge_robustness(dataset_label_list, model_predict_label_list,
                                      repaired_model_predict_dataset_label_list,
                                      model_predict_countermeasure_label_path,
                                      repaired_model_predict_countermeasure_label_list,
                                      L_Describe_1, L_Describe_2, alpha=0.7, beta=0.3)
        self.robustness = str(robustness)
        self.ui.robustness_result.setText(self.robustness)
        self.ui.logtextBrowser.append("成功计算鲁棒性...")
        self.ui.logtextBrowser.append("*" * 50)


# class Login:
#     def __init__(self):
#         # Load the UI definition from the file
#
#         # Dynamically create a corresponding window object from the UI definition
#         # Note: The control object inside is also a property of the window object
#         # such as self.ui.button , self.ui.textEdit
#         self.ui = QUiLoader().load('login.ui')
#         self.ui.pushButton.clicked.connect(self.openMain)
#
#     def openMain(self):
#         self.ui = Main()
#         # self.ui = QUiLoader().load('main.ui')
#         self.ui.show()
#         # self.ui = Main()
#         # self.ui.show()


# Generate Results
class ResultTest:
    def __init__(self):
        # # 改
        # self.ui = QUiLoader().load('testresult.ui')
        # result_title = '序号' + "\t" + '种子图片' + "\t" + "\t" + "\t" + '生成图片' + "\t" + "\t" + "\t" + '原标签' + "\t" + "\t" + "\t" + '模型预测标签' + "\t" + "\t" + "\t" + '覆蓋率'
        # self.ui.genTextBrowser.append(result_title)
        # self.generate_csv = new_input + '/' + "steering1.csv"
        # with open(self.generate_csv, 'r') as csvfile:
        #     reader = csv.reader(csvfile)
        #     head = next(reader)
        #     for row in reader:
        #         id = row[0]
        #         seed_image = row[1]
        #         generate_image = row[2]
        #         seed_label = row[3]
        #         predict_label = row[4]
        #         coverage = row[5]
        #         result = id + "\t" + seed_image + "\t" + "\t" + generate_image + "\t" + "\t" + seed_label + "\t" + "\t" + predict_label + "\t" + "\t" + coverage
        #         self.ui.genTextBrowser.append(result)

        self.ui = QUiLoader().load('testresult.ui')
        result_title = '序号' + "\t" + '种子图片' + "\t" + "\t" + "\t" + '生成图片' + "\t" + "\t" + "\t" + '标签'
        self.ui.genTextBrowser.append(result_title)
        self.generate_csv = new_input + '/' + "steering.csv"
        with open(self.generate_csv, 'r') as csvfile:
            reader = csv.reader(csvfile)
            head = next(reader)
            for row in reader:
                id = row[0]
                seed_image = row[1]
                generate_image = row[2]
                label = row[3]
                result = id + "\t" + seed_image + "\t" + "\t" + generate_image + "\t" + "\t" + label
                self.ui.genTextBrowser.append(result)

    def show(self):
        self.ui.show()


# The optimization results show
class Optimize:
    def __init__(self):
        self.ui = QUiLoader().load('optimize.ui')
        self.ui.dataResult.clicked.connect(self.openData)
        self.ui.chartResult.clicked.connect(self.openChart)
        self.ui.reportResult.clicked.connect(self.openReport)

    # Show the data
    def openData(self):
        self.data = DataForm()
        self.data.show()

    def openChart(self):
        self.chart = ChartForm()
        self.chart.show()

    def openReport(self):
        self.report = ReportForm()
        self.report.show()

    def show(self):
        self.ui.show()


class DataForm:
    def __init__(self):
        self.ui = QUiLoader().load('dataform.ui')

    def show(self):
        self.ui.show()


class ChartForm:
    def __init__(self):
        self.ui = QUiLoader().load('chartform.ui')

    def show(self):
        self.ui.show()


class ReportForm:
    def __init__(self):
        self.ui = QUiLoader().load('reportform.ui')

    def show(self):
        self.ui.show()


# New added
class Login:
    def __init__(self):
        # Load the UI definition from the file

        # Dynamically create a corresponding window object from the UI definition
        # Note: The control object inside is also a property of the window object
        # such as self.ui.button , self.ui.textEdit
        self.ui = QUiLoader().load('login_generate.ui')

        # from user_info import usr_info
        # usr_name = self.ui.lineEdit.text()
        # usr_pwd = self.ui.lineEdit_2.text()
        # flag = usr_info(usr_name, usr_pwd)
        # if flag:

        self.ui.pushButton.clicked.connect(self.openMain)

    # def usr_login(self):
    #     usr_name = self.ui.lineEdit.text()
    #     usr_pwd = self.ui.lineEdit_2.text()
    #     try:
    #         with open('usrs_info,pickle', 'rb') as usr_file:
    #             usrs_info = pickle.load(usr_file)
    #     except FileNotFoundError:
    #         with open('usrs_info', 'wb') as usr_file:
    #             usrs_info = {"admin":"admin"}
    #             pickle.dump(usrs_info, usr_file)
    #     if usr_name in usrs_info:
    #         if usr_pwd == usrs_info[usr_name]:
    #             self.ui = Main()
    #             self.ui.show()
    #         else:
    #             self.ui.messagebox.showinfo(message = '密码错误！')
    #     else:
    #         is_sign_up = self.ui.messagebox.askyeson("尚未注册！")

    # if is_sign_up:
    #     self.usr_sign_up()

    def openMain(self):
        if check_user(self.ui.lineEdit.text(), self.ui.lineEdit_2.text()):
            self.ui = Main()

        # self.ui = QUiLoader().load('main.ui')
            self.ui.show()
        # self.ui = Main()
        # self.ui.show()


if __name__ == '__main__':
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)
    app = QApplication([])
    app.setWindowIcon(QIcon('./log.ico'))
    login = Login()

    login.ui.show()

    app.exec_()

    # s = Main()
    # s.getImageCover()
