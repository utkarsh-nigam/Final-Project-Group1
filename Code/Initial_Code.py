import sys,os
os.chdir("/Users/utkarshvirendranigam/Desktop/Homework/Project")
#from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QAction, QComboBox, QLabel, QGridLayout, QCheckBox, QGroupBox
from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget, QPushButton, QAction, QComboBox, QLabel,
                             QGridLayout, QCheckBox, QGroupBox, QVBoxLayout, QHBoxLayout, QLineEdit, QPlainTextEdit)

from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import Qt

from scipy import interp
from itertools import cycle


from PyQt5.QtWidgets import QDialog, QVBoxLayout, QSizePolicy, QMessageBox

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import polyfit

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Libraries to display decision tree
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
import webbrowser

import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

import random
import seaborn as sns

#%%-----------------------------------------------------------------------
import os
os.environ["PATH"] += os.pathsep + 'C:\\Program Files (x86)\\graphviz-2.38\\release\\bin'
#%%-----------------------------------------------------------------------


#::--------------------------------
# Deafault font size for all the windows
#::--------------------------------
font_size_window = 'font-size:15px'

class RandomForest(QMainWindow):
    #::--------------------------------------------------------------------------------
    # Implementation of Random Forest Classifier using the happiness dataset
    # the methods in this class are
    #       _init_ : initialize the class
    #       initUi : creates the canvas and all the elements in the canvas
    #       update : populates the elements of the canvas base on the parametes
    #               chosen by the user
    #::---------------------------------------------------------------------------------
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(RandomForest, self).__init__()
        self.Title = "Random Forest Classifier"
        self.initUi()

    def initUi(self):
        #::-----------------------------------------------------------------
        #  Create the canvas and all the element to create a dashboard with
        #  all the necessary elements to present the results from the algorithm
        #  The canvas is divided using a  grid loyout to facilitate the drawing
        #  of the elements
        #::-----------------------------------------------------------------

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.main_widget = QWidget(self)

        self.layout = QGridLayout(self.main_widget)

        self.groupBox1 = QGroupBox('Random Forest Features')
        self.groupBox1Layout= QGridLayout()   # Grid
        self.groupBox1.setLayout(self.groupBox1Layout)

        # We create a checkbox of each Features
        self.feature0 = QCheckBox(features_list[0],self)
        self.feature1 = QCheckBox(features_list[1],self)
        self.feature2 = QCheckBox(features_list[2], self)
        self.feature3 = QCheckBox(features_list[3], self)
        self.feature4 = QCheckBox(features_list[4],self)
        self.feature5 = QCheckBox(features_list[5],self)
        self.feature6 = QCheckBox(features_list[6], self)
        self.feature7 = QCheckBox(features_list[7], self)
        self.feature8 = QCheckBox(features_list[8], self)
        self.feature9 = QCheckBox(features_list[9], self)
        self.feature10 = QCheckBox(features_list[10], self)
        self.feature11 = QCheckBox(features_list[11], self)
        self.feature12 = QCheckBox(features_list[12], self)
        self.feature13 = QCheckBox(features_list[13], self)
        self.feature14 = QCheckBox(features_list[14], self)
        self.feature15 = QCheckBox(features_list[15], self)
        self.feature16 = QCheckBox(features_list[16], self)
        self.feature17 = QCheckBox(features_list[17], self)
        self.feature18 = QCheckBox(features_list[18], self)
        self.feature19 = QCheckBox(features_list[19], self)
        self.feature0.setChecked(True)
        self.feature1.setChecked(True)
        self.feature2.setChecked(True)
        self.feature3.setChecked(True)
        self.feature4.setChecked(True)
        self.feature5.setChecked(True)
        self.feature6.setChecked(True)
        self.feature7.setChecked(True)
        self.feature8.setChecked(True)
        self.feature9.setChecked(True)
        self.feature10.setChecked(True)
        self.feature11.setChecked(True)
        self.feature12.setChecked(True)
        self.feature13.setChecked(True)
        self.feature14.setChecked(True)
        self.feature15.setChecked(True)
        self.feature16.setChecked(True)
        self.feature17.setChecked(True)
        self.feature18.setChecked(True)
        self.feature19.setChecked(True)

        self.lblPercentTest = QLabel('Percentage for Test :')
        self.lblPercentTest.adjustSize()

        self.txtPercentTest = QLineEdit(self)
        self.txtPercentTest.setText("20")

        self.btnExecute = QPushButton("Run Model")
        self.btnExecute.clicked.connect(self.update)

        self.groupBox1Layout.addWidget(self.btnExecute, 0, 0)
        self.groupBox1Layout.addWidget(self.lblPercentTest, 1, 0)
        self.groupBox1Layout.addWidget(self.txtPercentTest, 1, 1)
        self.groupBox1Layout.addWidget(self.feature0,2,0)
        self.groupBox1Layout.addWidget(self.feature1,2,1)
        self.groupBox1Layout.addWidget(self.feature2,3,0)
        self.groupBox1Layout.addWidget(self.feature3,3,1)
        self.groupBox1Layout.addWidget(self.feature4,4,0)
        self.groupBox1Layout.addWidget(self.feature5,4,1)
        self.groupBox1Layout.addWidget(self.feature6,5,0)
        self.groupBox1Layout.addWidget(self.feature7,5,1)
        self.groupBox1Layout.addWidget(self.feature8, 6, 0)
        self.groupBox1Layout.addWidget(self.feature9, 6, 1)
        self.groupBox1Layout.addWidget(self.feature10, 7, 0)
        self.groupBox1Layout.addWidget(self.feature11, 7, 1)
        self.groupBox1Layout.addWidget(self.feature12, 8, 0)
        self.groupBox1Layout.addWidget(self.feature13, 8, 1)
        self.groupBox1Layout.addWidget(self.feature14, 9, 0)
        self.groupBox1Layout.addWidget(self.feature15, 9, 1)
        self.groupBox1Layout.addWidget(self.feature16, 10, 0)
        self.groupBox1Layout.addWidget(self.feature17, 10, 1)
        self.groupBox1Layout.addWidget(self.feature18, 11, 0)
        self.groupBox1Layout.addWidget(self.feature19, 11, 1)


        self.groupBox2 = QGroupBox('Results from the mdel')
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)

        self.lblResults = QLabel('Results:')
        self.lblResults.adjustSize()
        self.txtResults = QPlainTextEdit()
        self.lblAccuracy = QLabel('Accuracy:')
        self.txtAccuracy = QLineEdit()

        self.groupBox2Layout.addWidget(self.lblResults)
        self.groupBox2Layout.addWidget(self.txtResults)
        self.groupBox2Layout.addWidget(self.lblAccuracy)
        self.groupBox2Layout.addWidget(self.txtAccuracy)

        #::--------------------------------------
        # Graphic 1 : Confusion Matrix
        #::--------------------------------------

        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes=[self.ax1]
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas.updateGeometry()

        self.groupBoxG1 = QGroupBox('Confusion Matrix')
        self.groupBoxG1Layout= QVBoxLayout()
        self.groupBoxG1.setLayout(self.groupBoxG1Layout)

        self.groupBoxG1Layout.addWidget(self.canvas)

        #::---------------------------------------
        # Graphic 2 : ROC Curve
        #::---------------------------------------

        self.fig2 = Figure()
        self.ax2 = self.fig2.add_subplot(111)
        self.axes2 = [self.ax2]
        self.canvas2 = FigureCanvas(self.fig2)

        self.canvas2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas2.updateGeometry()

        self.groupBoxG2 = QGroupBox('ROC Curve')
        self.groupBoxG2Layout = QVBoxLayout()
        self.groupBoxG2.setLayout(self.groupBoxG2Layout)

        self.groupBoxG2Layout.addWidget(self.canvas2)

        #::-------------------------------------------
        # Graphic 3 : Importance of Features
        #::-------------------------------------------

        self.fig3 = Figure()
        self.ax3 = self.fig3.add_subplot(111)
        self.axes3 = [self.ax3]
        self.canvas3 = FigureCanvas(self.fig3)

        self.canvas3.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas3.updateGeometry()

        self.groupBoxG3 = QGroupBox('Importance of Features')
        self.groupBoxG3Layout = QVBoxLayout()
        self.groupBoxG3.setLayout(self.groupBoxG3Layout)
        self.groupBoxG3Layout.addWidget(self.canvas3)

        #::--------------------------------------------
        # Graphic 4 : ROC Curve by class
        #::--------------------------------------------

        self.fig4 = Figure()
        self.ax4 = self.fig4.add_subplot(111)
        self.axes4 = [self.ax4]
        self.canvas4 = FigureCanvas(self.fig4)

        self.canvas4.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas4.updateGeometry()

        self.groupBoxG4 = QGroupBox('ROC Curve by Class')
        self.groupBoxG4Layout = QVBoxLayout()
        self.groupBoxG4.setLayout(self.groupBoxG4Layout)
        self.groupBoxG4Layout.addWidget(self.canvas4)

        #::-------------------------------------------------
        # End of graphs
        #::-------------------------------------------------

        self.layout.addWidget(self.groupBox1,0,0)
        self.layout.addWidget(self.groupBoxG1,0,1)
        self.layout.addWidget(self.groupBoxG3, 0, 2)
        self.layout.addWidget(self.groupBox2,1,0)
        self.layout.addWidget(self.groupBoxG2,1,1)
        self.layout.addWidget(self.groupBoxG4,1,2)

        self.setCentralWidget(self.main_widget)
        self.resize(1100, 700)
        self.show()

    def update(self):
        '''
        Random Forest Classifier
        We pupulate the dashboard using the parametres chosen by the user
        The parameters are processed to execute in the skit-learn Random Forest algorithm
          then the results are presented in graphics and reports in the canvas
        :return:None
        '''

        # processing the parameters

        self.list_corr_features = pd.DataFrame([])
        if self.feature0.isChecked():
            if len(self.list_corr_features)==0:
                self.list_corr_features = attr_data[features_list[0]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, attr_data[features_list[0]]],axis=1)

        if self.feature1.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = attr_data[features_list[1]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, attr_data[features_list[1]]],axis=1)

        if self.feature2.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = attr_data[features_list[2]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, attr_data[features_list[2]]],axis=1)

        if self.feature3.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = attr_data[features_list[3]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, attr_data[features_list[3]]],axis=1)

        if self.feature4.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = attr_data[features_list[4]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, attr_data[features_list[4]]],axis=1)

        if self.feature5.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = attr_data[features_list[5]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, attr_data[features_list[5]]],axis=1)

        if self.feature6.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = attr_data[features_list[6]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, attr_data[features_list[6]]],axis=1)

        if self.feature7.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = attr_data[features_list[7]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, attr_data[features_list[7]]],axis=1)

        if self.feature8.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = attr_data[features_list[8]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, attr_data[features_list[8]]],axis=1)

        if self.feature9.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = attr_data[features_list[9]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, attr_data[features_list[9]]],axis=1)

        if self.feature10.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = attr_data[features_list[10]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, attr_data[features_list[10]]], axis=1)

        if self.feature11.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = attr_data[features_list[11]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, attr_data[features_list[11]]], axis=1)

        if self.feature12.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = attr_data[features_list[12]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, attr_data[features_list[12]]], axis=1)

        if self.feature13.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = attr_data[features_list[13]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, attr_data[features_list[13]]], axis=1)

        if self.feature14.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = attr_data[features_list[14]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, attr_data[features_list[14]]], axis=1)

        if self.feature15.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = attr_data[features_list[15]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, attr_data[features_list[15]]], axis=1)

        if self.feature16.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = attr_data[features_list[16]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, attr_data[features_list[16]]], axis=1)

        if self.feature17.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = attr_data[features_list[17]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, attr_data[features_list[17]]], axis=1)

        if self.feature18.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = attr_data[features_list[18]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, attr_data[features_list[18]]], axis=1)

        if self.feature19.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = attr_data[features_list[19]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, attr_data[features_list[19]]], axis=1)


        vtest_per = float(self.txtPercentTest.text())

        # Clear the graphs to populate them with the new information

        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        self.txtResults.clear()
        self.txtResults.setUndoRedoEnabled(False)

        vtest_per = vtest_per / 100

        # Assign the X and y to run the Random Forest Classifier

        X_dt =  self.list_corr_features
        #temp_X_dt=X_dt.copy()
        y_dt = attr_data[target_variable]
        X_columns=X_dt.columns.tolist()
        labelencoder_columns= list(set(X_columns) & set(label_encoder_variables))
        one_hot_encoder_columns=list(set(X_columns) & set(hot_encoder_variables))
        #print(labelencoder_columns)
        #print(one_hot_encoder_columns)
        class_le = LabelEncoder()
        class_ohe=OneHotEncoder()
        #X_dt[labelencoder_columns] = class_le.fit_transform(X_dt[labelencoder_columns])
        temp = X_columns.copy()
        for ohe_val in one_hot_encoder_columns:
            temp.remove(ohe_val)
        temp_X_dt=X_dt[temp]
        for le_val in labelencoder_columns:
            temp_X_dt[le_val] = class_le.fit_transform(temp_X_dt[le_val])
        X_dt=pd.concat((temp_X_dt,pd.get_dummies(X_dt[one_hot_encoder_columns])),1)
        # fit and transform the class

        y_dt = class_le.fit_transform(y_dt)

        # split the dataset into train and test

        X_train, X_test, y_train, y_test = train_test_split(X_dt, y_dt, test_size=vtest_per, random_state=500)

        # perform training with entropy.
        # Decision tree with entropy

        #specify random forest classifier
        self.clf_rf = RandomForestClassifier(n_estimators=100, random_state=500)

        # perform training
        self.clf_rf.fit(X_train, y_train)

        #-----------------------------------------------------------------------

        # predicton on test using all features
        y_pred = self.clf_rf.predict(X_test)
        y_pred_score = self.clf_rf.predict_proba(X_test)


        # confusion matrix for RandomForest
        conf_matrix = confusion_matrix(y_test, y_pred)

        # clasification report

        self.ff_class_rep = classification_report(y_test, y_pred)
        self.txtResults.appendPlainText(self.ff_class_rep)

        # accuracy score

        self.ff_accuracy_score = accuracy_score(y_test, y_pred) * 100
        self.txtAccuracy.setText(str(self.ff_accuracy_score))

        #::------------------------------------
        ##  Ghaph1 :
        ##  Confusion Matrix
        #::------------------------------------
        class_names1 = ['','No', 'Yes']

        self.ax1.matshow(conf_matrix, cmap= plt.cm.get_cmap('Blues', 14))
        self.ax1.set_yticklabels(class_names1)
        self.ax1.set_xticklabels(class_names1,rotation = 90)
        self.ax1.set_xlabel('Predicted label')
        self.ax1.set_ylabel('True label')

        for i in range(len(class_names)):
            for j in range(len(class_names)):
                y_pred_score = self.clf_rf.predict_proba(X_test)
                self.ax1.text(j, i, str(conf_matrix[i][j]))

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

        ## End Graph1 -- Confusion Matrix

        #::----------------------------------------
        ## Graph 2 - ROC Curve
        #::----------------------------------------
        #y_test_bin = label_binarize(y_test, classes=[0, 1])
        #print(pd.get_dummies(y_test))
        #print(pd.get_dummies(y_test).to_numpy())
        y_test_bin=pd.get_dummies(y_test).to_numpy()
        n_classes = y_test_bin.shape[1]

        #From the sckict learn site
        #https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        #print(pd.get_dummies(y_test).to_numpy().ravel())

        #print("\n\n********************************\n\n")
        #print(y_pred_score.ravel())
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_pred_score.ravel())

        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        lw = 2
        self.ax2.plot(fpr[1], tpr[1], color='darkorange',
                      lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
        self.ax2.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        self.ax2.set_xlim([0.0, 1.0])
        self.ax2.set_ylim([0.0, 1.05])
        self.ax2.set_xlabel('False Positive Rate')
        self.ax2.set_ylabel('True Positive Rate')
        self.ax2.set_title('ROC Curve Random Forest')
        self.ax2.legend(loc="lower right")

        self.fig2.tight_layout()
        self.fig2.canvas.draw_idle()
        ######################################
        # Graph - 3 Feature Importances
        #####################################
        # get feature importances
        importances = self.clf_rf.feature_importances_

        # convert the importances into one-dimensional 1darray with corresponding df column names as axis labels
        f_importances = pd.Series(importances, X_dt.columns)

        # sort the array in descending order of the importances
        f_importances.sort_values(ascending=False, inplace=True)
        f_importances=f_importances[0:10]
        X_Features = f_importances.index
        y_Importance = list(f_importances)

        self.ax3.barh(X_Features, y_Importance )
        self.ax3.set_aspect('auto')

        # show the plot
        self.fig3.tight_layout()
        self.fig3.canvas.draw_idle()

        #::-----------------------------------------------------
        # Graph 4 - ROC Curve by Class
        #::-----------------------------------------------------
        str_classes= ['HP','MEH','LOH','NH']
        colors = cycle(['magenta', 'darkorange', 'green', 'blue'])
        for i, color in zip(range(n_classes), colors):
            self.ax4.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='{0} (area = {1:0.2f})'
                           ''.format(str_classes[i], roc_auc[i]))

        self.ax4.plot([0, 1], [0, 1], 'k--', lw=lw)
        self.ax4.set_xlim([0.0, 1.0])
        self.ax4.set_ylim([0.0, 1.05])
        self.ax4.set_xlabel('False Positive Rate')
        self.ax4.set_ylabel('True Positive Rate')
        self.ax4.set_title('ROC Curve by Class')
        self.ax4.legend(loc="lower right")

        # show the plot
        self.fig4.tight_layout()
        self.fig4.canvas.draw_idle()

        #::-----------------------------
        # End of graph 4  - ROC curve by class
        #::-----------------------------


class PlotCanvas(FigureCanvas):
    #::----------------------------------------------------------
    # creates a figure on the canvas
    # later on this element will be used to draw a histogram graph
    #::----------------------------------------------------------
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def plot(self):
        self.ax = self.figure.add_subplot(111)

class CanvasWindow(QMainWindow):
    #::----------------------------------
    # Creates a canvaas containing the plot for the initial analysis
    #;;----------------------------------
    def __init__(self, parent=None):
        super(CanvasWindow, self).__init__(parent)

        self.left = 200
        self.top = 200
        self.Title = 'Distribution'
        self.width = 500
        self.height = 500
        self.initUI()

    def initUI(self):

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.setGeometry(self.left, self.top, self.width, self.height)

        self.m = PlotCanvas(self, width=5, height=4)
        self.m.move(0, 30)

class App(QMainWindow):
    #::-------------------------------------------------------
    # This class creates all the elements of the application
    #::-------------------------------------------------------

    def __init__(self):
        super().__init__()
        self.left = 200
        self.top = 200
        self.width = 1000
        self.height = 500
        self.Title = 'Attrition Predictor'
        self.initUI()

    def initUI(self):
        #::-------------------------------------------------
        # Creates the menu and the items
        #::-------------------------------------------------
        self.setWindowTitle(self.Title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        #::-----------------------------
        # Create the menu bar
        # and three items for the menu, File, EDA Analysis and ML Models
        #::-----------------------------
        mainMenu = self.menuBar()
        mainMenu.setStyleSheet('background-color: lightblue')

        fileMenu = mainMenu.addMenu('File')
        EDAMenu = mainMenu.addMenu('EDA Analysis')
        MLModelMenu = mainMenu.addMenu('ML Models')

        #::--------------------------------------
        # Exit application
        # Creates the actions for the fileMenu item
        #::--------------------------------------

        exitButton = QAction(QIcon('enter.png'), 'Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.setStatusTip('Exit application')
        exitButton.triggered.connect(self.close)

        fileMenu.addAction(exitButton)

        #::----------------------------------------
        # EDA analysis
        # Creates the actions for the EDA Analysis item
        # Initial Assesment : Histogram about the level of happiness in 2017
        # Happiness Final : Presents the correlation between the index of happiness and a feature from the datasets.
        # Correlation Plot : Correlation plot using all the dims in the datasets
        #::----------------------------------------

        EDA1Button = QAction(QIcon('analysis.png'),'Initial Assesment', self)
        EDA1Button.setStatusTip('Presents the initial datasets')
        EDA1Button.triggered.connect(self.EDA1)
        EDAMenu.addAction(EDA1Button)

        EDA2Button = QAction(QIcon('analysis.png'), 'Happiness Final', self)
        EDA2Button.setStatusTip('Final Happiness Graph')
        EDA2Button.triggered.connect(self.EDA2)
        EDAMenu.addAction(EDA2Button)

        EDA4Button = QAction(QIcon('analysis.png'), 'Correlation Plot', self)
        EDA4Button.setStatusTip('Features Correlation Plot')
        EDA4Button.triggered.connect(self.EDA4)
        EDAMenu.addAction(EDA4Button)

        #::--------------------------------------------------
        # ML Models for prediction
        # There are two models
        #       Decision Tree
        #       Random Forest
        #::--------------------------------------------------
        # Decision Tree Model
        #::--------------------------------------------------
        MLModel1Button =  QAction(QIcon(), 'Decision Tree Entropy', self)
        MLModel1Button.setStatusTip('ML algorithm with Entropy ')
        MLModel1Button.triggered.connect(self.MLDT)

        #::------------------------------------------------------
        # Random Forest Classifier
        #::------------------------------------------------------
        MLModel2Button = QAction(QIcon(), 'Random Forest Classifier', self)
        MLModel2Button.setStatusTip('Random Forest Classifier ')
        MLModel2Button.triggered.connect(self.MLRF)

        MLModelMenu.addAction(MLModel1Button)
        MLModelMenu.addAction(MLModel2Button)

        self.dialogs = list()

    def EDA1(self):
        #::------------------------------------------------------
        # Creates the histogram
        # The X variable contains the happiness.score
        # X was populated in the method data_happiness()
        # at the start of the application
        #::------------------------------------------------------
        dialog = CanvasWindow(self)
        dialog.m.plot()
        dialog.m.ax.hist(X, bins=12, facecolor='green', alpha=0.5)
        dialog.m.ax.set_title('Frequency of Happiness Year 2017')
        dialog.m.ax.set_xlabel("Level of Happiness")
        dialog.m.ax.set_ylabel("Number of Countries")
        dialog.m.ax.grid(True)
        dialog.m.draw()
        self.dialogs.append(dialog)
        dialog.show()

    def EDA2(self):
        #::---------------------------------------------------------
        # This function creates an instance of HappinessGraphs class
        # This class creates a graph using the features in the dataset
        # happiness vrs the score of happiness
        #::---------------------------------------------------------
        dialog = HappinessGraphs()
        self.dialogs.append(dialog)
        dialog.show()

    def EDA4(self):
        #::----------------------------------------------------------
        # This function creates an instance of the CorrelationPlot class
        #::----------------------------------------------------------
        dialog = CorrelationPlot()
        self.dialogs.append(dialog)
        dialog.show()

    def MLDT(self):
        #::-----------------------------------------------------------
        # This function creates an instance of the DecisionTree class
        # This class presents a dashboard for a Decision Tree Algorithm
        # using the happiness dataset
        #::-----------------------------------------------------------
        dialog = DecisionTree()
        self.dialogs.append(dialog)
        dialog.show()

    def MLRF(self):
        #::-------------------------------------------------------------
        # This function creates an instance of the Random Forest Classifier Algorithm
        # using the happiness dataset
        #::-------------------------------------------------------------
        dialog = RandomForest()
        self.dialogs.append(dialog)
        dialog.show()

def main():
    #::-------------------------------------------------
    # Initiates the application
    #::-------------------------------------------------
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    ex = App()
    ex.show()
    sys.exit(app.exec_())


def attrition_data():
    #::--------------------------------------------------
    # Loads the dataset 2017.csv ( Index of happiness and esplanatory variables original dataset)
    # Loads the dataset final_happiness_dataset (index of happiness
    # and explanatory variables which are already preprocessed)
    # Populates X,y that are used in the classes above
    #::--------------------------------------------------
    global happiness
    global attr_data
    global X
    global y
    global features_list
    global class_names
    global target_variable
    global label_encoder_variables
    global hot_encoder_variables
    #happiness = pd.read_csv('2017.csv')
    #X= happiness["Happiness.Score"]
    #y= happiness["Country"]
    attr_data = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
    all_columns = attr_data.columns.tolist()
    #print(all_columns)
    all_columns.remove("Attrition")
    features_list=all_columns.copy()
    #print(features_list)
    target_variable="Attrition"
    class_names = ['No', 'Yes']
    label_encoder_variables =["Education","JobLevel"]
    hot_encoder_variables=["BusinessTravel","Department","EducationField","Gender","JobRole","MaritalStatus","OverTime","StockOptionLevel"]

if __name__ == '__main__':
    #::------------------------------------
    # First reads the data then calls for the application
    #::------------------------------------
    attrition_data()
    main()
