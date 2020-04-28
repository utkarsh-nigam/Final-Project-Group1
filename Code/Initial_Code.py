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
from itertools import cycle, combinations
import random
import statsmodels.api as sm
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QSizePolicy, QFormLayout, QRadioButton, QScrollArea, QMessageBox
from PyQt5.QtGui import QPixmap

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import polyfit

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc, log_loss, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn import feature_selection
from sklearn import metrics
from sklearn.preprocessing import label_binarize

# Libraries to display decision tree
from pydotplus import graph_from_dot_data
import collections
#from sklearn.tree import export_graphviz
#import webbrowser

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


class VariableDistribution(QMainWindow):
    #::---------------------------------------------------------
    # This class crates a canvas with a plot to show the relation
    # from each feature in the dataset with the happiness score
    # methods
    #    _init_
    #   update
    #::---------------------------------------------------------
    send_fig = pyqtSignal(str)

    def __init__(self):
        #::--------------------------------------------------------
        # Crate a canvas with the layout to draw a dotplot
        # The layout sets all the elements and manage the changes
        # made on the canvas
        #::--------------------------------------------------------
        super(VariableDistribution, self).__init__()

        self.Title = "EDA: Variable Distribution"
        self.main_widget = QWidget(self)

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        self.axes=[self.ax]
        self.canvas = FigureCanvas(self.fig)


        self.canvas.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)

        self.canvas.updateGeometry()
        self.featuresList=personal_features
        self.dropdown1 = QComboBox()
        self.dropdown1.addItems(["Personal", "Organisation", "Commution"])
        self.dropdown1.currentIndexChanged.connect(self.updateCategory)
        self.dropdown2 = QComboBox()
        self.label = QLabel("A plot:")
        self.filter_data = QWidget(self)
        self.filter_data.layout = QGridLayout(self.filter_data)

        self.filter_data.layout.addWidget(QLabel("Choose Data Filter:"), 0, 0, 1, 1)

        self.filter_radio_button = QRadioButton("All Data")
        self.filter_radio_button.setChecked(True)
        self.filter_radio_button.filter = "All_Data"
        self.set_Filter = "All_Data"
        self.filter_radio_button.toggled.connect(self.onFilterClicked)
        self.filter_data.layout.addWidget(self.filter_radio_button, 0, 1, 1, 1)

        self.filter_radio_button = QRadioButton("Attrition: Yes")
        self.filter_radio_button.filter = "Yes"
        self.filter_radio_button.toggled.connect(self.onFilterClicked)
        self.filter_data.layout.addWidget(self.filter_radio_button, 0, 2, 1, 1)

        self.filter_radio_button = QRadioButton("Attrition: No")
        self.filter_radio_button.filter = "No"
        self.filter_radio_button.toggled.connect(self.onFilterClicked)
        self.filter_data.layout.addWidget(self.filter_radio_button, 0, 3, 1, 1)

        self.btnCreateGraph = QPushButton("Create Graph")
        self.btnCreateGraph.clicked.connect(self.update)
        self.filter_data.layout.addWidget(self.btnCreateGraph, 1, 0, 4, 1)


        self.groupBox1 = QGroupBox('Distribution')
        self.groupBox1Layout = QVBoxLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)
        #self.groupBox2.setMinimumSize(400, 50)
        self.groupBox1Layout.addWidget(self.canvas)

        self.groupBox2 = QGroupBox('Summary')
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)
        # self.groupBox2.setMinimumSize(400, 50)
        self.graph_summary = QPlainTextEdit()
        self.groupBox2Layout.addWidget(self.graph_summary)


        self.layout = QGridLayout(self.main_widget)
        self.layout.addWidget(QLabel("Select Feature Category:"),0,0,2,1)
        self.layout.addWidget(self.dropdown1,0,1,2,1)
        self.layout.addWidget(QLabel(""), 0, 2, 2, 1)
        self.layout.addWidget(QLabel("Select Features:"),0,3,2,1)
        self.layout.addWidget(self.dropdown2,0,4,2,1)
        self.layout.addWidget(QLabel(""), 0, 5, 2, 1)
        self.layout.addWidget(QLabel("Choose Data Filter:"), 0, 6, 2, 1)
        self.layout.addWidget(self.filter_data,0,7,2,2)
        self.layout.addWidget(self.groupBox1,2,0,5,5)
        self.layout.addWidget(QLabel(""), 2, 5, 5, 1)
        self.layout.addWidget(self.groupBox2, 2, 6, 5, 3)

        self.setCentralWidget(self.main_widget)
        self.resize(1400, 700)
        self.show()
        self.updateCategory()

    def updateCategory(self):
        self.dropdown2.clear()
        feature_category = self.dropdown1.currentText()
        if(feature_category=="Personal"):
            self.featuresList=list(set(continuous_features) & set(personal_features))
        elif (feature_category == "Organisation"):
            self.featuresList = list(set(continuous_features) & set(organisation_features))
        elif (feature_category == "Commution"):
            self.featuresList = list(set(continuous_features) & set(commution_features))
        del feature_category
        self.dropdown2.addItems(self.featuresList)
        del self.featuresList

    def onFilterClicked(self):
        self.filter_radio_button = self.sender()
        if self.filter_radio_button.isChecked():
            self.set_Filter=self.filter_radio_button.filter
            self.update()

    def update(self):
        #::--------------------------------------------------------
        # This method executes each time a change is made on the canvas
        # containing the elements of the graph
        # The purpose of the method es to draw a dot graph using the
        # score of happiness and the feature chosen the canvas
        #::--------------------------------------------------------
        colors=["b", "r", "g", "y", "k", "c"]
        self.ax.clear()
        cat1 = self.dropdown2.currentText()
        if (self.set_Filter=="Yes" or self.set_Filter=="No"):
            self.filtered_data=attr_data.copy()
            self.filtered_data = self.filtered_data[self.filtered_data["Attrition"]==self.set_Filter]
        else:
            self.filtered_data = attr_data.copy()

        self.ax.hist(self.filtered_data[cat1], bins=10, facecolor='green', alpha=0.5)
        self.ax.set_title(cat1)
        self.ax.set_xlabel(cat1)
        self.ax.set_ylabel("Count")
        self.ax.grid(True)
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()
        del cat1
        del self.filtered_data


class VariableRelation(QMainWindow):
    #::---------------------------------------------------------
    # This class crates a canvas with a plot to show the relation
    # from each feature in the dataset with the happiness score
    # methods
    #    _init_
    #   update
    #::---------------------------------------------------------
    send_fig = pyqtSignal(str)

    def __init__(self):
        #::--------------------------------------------------------
        # Crate a canvas with the layout to draw a dotplot
        # The layout sets all the elements and manage the changes
        # made on the canvas
        #::--------------------------------------------------------
        super(VariableRelation, self).__init__()

        self.Title = "EDA: Variable Relation"
        self.main_widget = QWidget(self)

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        self.axes=[self.ax]
        self.canvas = FigureCanvas(self.fig)


        self.canvas.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)

        self.canvas.updateGeometry()
        self.featuresList1 = personal_features
        self.featuresList2 = personal_features



        self.filterBox1 = QGroupBox('Feature 1')
        self.filterBox1Layout = QGridLayout()
        self.filterBox1.setLayout(self.filterBox1Layout)

        self.dropdown1 = QComboBox()
        self.dropdown1.addItems(["Personal", "Organisation", "Commution", "Satisfaction"])
        self.dropdown1.currentIndexChanged.connect(self.updateCategory1)
        self.dropdown2 = QComboBox()
        self.dropdown2.addItems(self.featuresList1)
        self.filterBox1Layout.addWidget(QLabel("Select Feature Category:"),0,0)
        self.filterBox1Layout.addWidget(self.dropdown1,0,1)
        self.filterBox1Layout.addWidget(QLabel("Select Feature:"),1,0)
        self.filterBox1Layout.addWidget(self.dropdown2,1,1)




        self.filterBox2 = QGroupBox('Feature 2')
        self.filterBox2Layout = QGridLayout()
        self.filterBox2.setLayout(self.filterBox2Layout)

        self.dropdown3 = QComboBox()
        self.dropdown3.addItems(["Personal", "Organisation", "Commution", "Satisfaction"])
        self.dropdown3.currentIndexChanged.connect(self.updateCategory2)
        self.dropdown4 = QComboBox()
        self.dropdown4.addItems(self.featuresList2)
        self.filterBox2Layout.addWidget(QLabel("Select Feature Category:"),0,0)
        self.filterBox2Layout.addWidget(self.dropdown3,0,1)
        self.filterBox2Layout.addWidget(QLabel("Select Feature:"),1,0)
        self.filterBox2Layout.addWidget(self.dropdown4,1,1)




        self.filter_data = QWidget(self)
        self.filter_data.layout = QGridLayout(self.filter_data)

        self.filter_data.layout.addWidget(QLabel("Choose Data Filter:"), 0, 0, 1, 1)

        self.filter_radio_button = QRadioButton("All Data")
        self.filter_radio_button.setChecked(True)
        self.filter_radio_button.filter = "All_Data"
        self.set_Filter="All_Data"
        self.filter_radio_button.toggled.connect(self.onFilterClicked)
        self.filter_data.layout.addWidget(self.filter_radio_button, 0, 1,1,1)

        self.filter_radio_button = QRadioButton("Attrition: Yes")
        self.filter_radio_button.filter = "Yes"
        self.filter_radio_button.toggled.connect(self.onFilterClicked)
        self.filter_data.layout.addWidget(self.filter_radio_button, 0, 2,1,1)

        self.filter_radio_button = QRadioButton("Attrition: No")
        self.filter_radio_button.filter = "No"
        self.filter_radio_button.toggled.connect(self.onFilterClicked)
        self.filter_data.layout.addWidget(self.filter_radio_button, 0, 3,1,1)

        self.btnCreateGraph = QPushButton("Create Graph")
        self.btnCreateGraph.clicked.connect(self.update)
        self.filter_data.layout.addWidget(self.btnCreateGraph, 1, 0, 4, 1)






        self.groupBox1 = QGroupBox('Feature Relation')
        self.groupBox1Layout = QVBoxLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)
        #self.groupBox2.setMinimumSize(400, 50)
        self.groupBox1Layout.addWidget(self.canvas)

        self.groupBox2 = QGroupBox('Summary')
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)
        # self.groupBox2.setMinimumSize(400, 50)
        self.graph_summary = QPlainTextEdit()
        self.groupBox2Layout.addWidget(self.graph_summary)


        self.layout = QGridLayout(self.main_widget)
        self.layout.addWidget(self.filterBox1,0,0,2,2)
        self.layout.addWidget(QLabel(""), 0, 2, 2, 1)
        self.layout.addWidget(self.filterBox2,0,3,2,2)
        self.layout.addWidget(QLabel(""), 0, 5, 2, 1)
        self.layout.addWidget(self.filter_data,0,6,2,3)
        self.layout.addWidget(self.groupBox1,2,0,6,5)
        self.layout.addWidget(QLabel(""), 2, 5, 6, 1)
        self.layout.addWidget(self.groupBox2, 2, 6, 6, 3)


        self.setCentralWidget(self.main_widget)
        self.resize(1400, 700)
        self.show()
        self.updateCategory1()
        self.updateCategory2()


    def updateCategory1(self):
        self.dropdown2.clear()
        feature_category1 = self.dropdown1.currentText()
        if(feature_category1=="Personal"):
            self.featuresList1=personal_features
        elif (feature_category1 == "Organisation"):
            self.featuresList1 = organisation_features
        elif (feature_category1 == "Commution"):
            self.featuresList1 = commution_features
        elif (feature_category1 == "Satisfaction"):
            self.featuresList1 = satisfaction_features
        self.dropdown2.addItems(self.featuresList1)

    def updateCategory2(self):
        self.dropdown4.clear()
        feature_category2 = self.dropdown3.currentText()
        if (feature_category2 == "Personal"):
            self.featuresList2 = personal_features
        elif (feature_category2 == "Organisation"):
            self.featuresList2 = organisation_features
        elif (feature_category2 == "Commution"):
            self.featuresList2 = commution_features
        elif (feature_category2 == "Satisfaction"):
            self.featuresList2 = satisfaction_features
        self.dropdown4.addItems(self.featuresList2)

    def onFilterClicked(self):
        self.filter_radio_button = self.sender()
        if self.filter_radio_button.isChecked():
            self.set_Filter=self.filter_radio_button.filter
            self.update()

    def update(self):
        #::--------------------------------------------------------
        # This method executes each time a change is made on the canvas
        # containing the elements of the graph
        # The purpose of the method es to draw a dot graph using the
        # score of happiness and the feature chosen the canvas
        #::--------------------------------------------------------
        colors=["b", "r", "g", "y", "k", "c"]
        self.ax.clear()
        if (self.set_Filter=="Yes" or self.set_Filter=="No"):
            self.filtered_data=attr_data.copy()
            self.filtered_data = self.filtered_data[self.filtered_data["Attrition"]==self.set_Filter]
        else:
            self.filtered_data = attr_data.copy()

        graph_feature1 = self.dropdown2.currentText()
        graph_feature2 = self.dropdown4.currentText()

        if((graph_feature1 in categorical_features) and (graph_feature2 in categorical_features)):
            graph_data=self.filtered_data[[graph_feature1,graph_feature2]]
            graph_data["Employee Count"]=1
            my_pt = pd.pivot_table(graph_data, index=[graph_feature1,graph_feature2], values=["Employee Count"], aggfunc=np.sum)
            my_pt = pd.DataFrame(my_pt.to_records())
            my_pt=my_pt.pivot(graph_feature1, graph_feature2, "Employee Count")
            my_pt=my_pt.fillna(0)
            #print(my_pt)
            #print(my_pt.index.values)
            #print(my_pt.columns.values)

            #class_names1 = ['', 'Happy', 'Med.Happy', 'Low.Happy', 'Not.Happy']
            class_names_x=[""]+my_pt.columns.values.tolist()
            class_names_y = [""] + my_pt.index.values.tolist()
            self.ax.matshow(my_pt, cmap=plt.cm.get_cmap('Blues', 14))
            self.ax.set_yticklabels(class_names_y)
            self.ax.set_xticklabels(class_names_x, rotation=90)
            self.ax.set_xlabel(graph_feature2)
            self.ax.set_ylabel(graph_feature1)
            #self.ax.set_title("Heat Map")
            my_np=my_pt.values
            for i in range(len(class_names_y)-1):
                for j in range(len(class_names_x)-1):
                    self.ax.text(j, i, str(my_np[i][j]))

        elif((graph_feature1 in continuous_features) and (graph_feature2 in continuous_features)):
            x_axis_data = self.filtered_data[graph_feature1]
            y_axis_data = self.filtered_data[graph_feature2]
            #print(x_axis_data)
            #print(y_axis_data)
            self.ax.scatter(x_axis_data, y_axis_data)
            b, m = polyfit(x_axis_data, y_axis_data, 1)
            self.ax.plot(x_axis_data, b + m * x_axis_data, '-', color="orange")

            vtitle = graph_feature2 + " Vs " + graph_feature1
            self.ax.set_title(vtitle)
            self.ax.set_xlabel(graph_feature1)
            self.ax.set_ylabel(graph_feature2)
            self.ax.grid(True)

        else:

            if(graph_feature1 in continuous_features):
                continuous_data=graph_feature1#self.filtered_data[graph_feature1]
                categorical_data=graph_feature2#self.filtered_data[graph_feature2]
            else:
                continuous_data = graph_feature2#self.filtered_data[graph_feature2]
                categorical_data = graph_feature1#self.filtered_data[graph_feature1]
            graph_data = self.filtered_data[[graph_feature1, graph_feature2]]
            my_pt = pd.pivot_table(graph_data, index=graph_data.index,columns=categorical_data, values=continuous_data,aggfunc=np.sum)
            my_pt = pd.DataFrame(my_pt.to_records())
            my_pt=my_pt.drop(columns=['index'])
            #my_pt = my_pt.fillna("")
            class_names_x = my_pt.columns.values.tolist()
            my_np = my_pt.values
            mask = ~np.isnan(my_np)
            my_np_2 = [d[m] for d, m in zip(my_np.T, mask.T)]
            class_names_x = my_pt.columns.values.tolist()
            self.ax.boxplot(my_np_2)
            self.ax.set_xlabel(categorical_data)
            self.ax.set_ylabel(continuous_data)
            self.ax.set_xticklabels(class_names_x)



        self.fig.tight_layout()
        self.fig.canvas.draw_idle()


class AttritionRelation(QMainWindow):
    #::---------------------------------------------------------
    # This class crates a canvas with a plot to show the relation
    # from each feature in the dataset with the happiness score
    # methods
    #    _init_
    #   update
    #::---------------------------------------------------------
    send_fig = pyqtSignal(str)

    def __init__(self):
        #::--------------------------------------------------------
        # Crate a canvas with the layout to draw a dotplot
        # The layout sets all the elements and manage the changes
        # made on the canvas
        #::--------------------------------------------------------
        super(AttritionRelation, self).__init__()

        self.Title = "EDA: Attrition Relation"
        self.main_widget = QWidget(self)

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(1,2,1)
        self.axes1=[self.ax1]
        self.ax2 = self.fig.add_subplot(1, 2, 2)
        self.axes2 = [self.ax2]
        self.canvas = FigureCanvas(self.fig)


        self.canvas.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)

        self.canvas.updateGeometry()
        self.featuresList=personal_features
        self.dropdown1 = QComboBox()
        self.dropdown1.addItems(["Personal", "Organisation", "Commution", "Satisfaction"])
        self.dropdown1.currentIndexChanged.connect(self.updateCategory)
        self.dropdown2 = QComboBox()
        self.label = QLabel("A plot:")
        self.filter_data = QWidget(self)
        self.filter_data.layout = QGridLayout(self.filter_data)
        self.btnCreateGraph = QPushButton("Create Graph")
        self.btnCreateGraph.clicked.connect(self.update)

        self.groupBox1 = QGroupBox('Relation between Attrition: Yes and No')
        self.groupBox1Layout = QVBoxLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)
        #self.groupBox2.setMinimumSize(400, 50)
        self.groupBox1Layout.addWidget(self.canvas)

        self.groupBox2 = QGroupBox('Summary')
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)
        # self.groupBox2.setMinimumSize(400, 50)
        self.graph_summary = QPlainTextEdit()
        self.groupBox2Layout.addWidget(self.graph_summary)


        self.layout = QGridLayout(self.main_widget)
        self.layout.addWidget(QLabel("Select Feature Category:"),0,0,1,1)
        self.layout.addWidget(self.dropdown1,0,1,1,1)
        self.layout.addWidget(QLabel(""), 0, 2, 1, 1)
        self.layout.addWidget(QLabel("Select Features:"),0,3,1,1)
        self.layout.addWidget(self.dropdown2,0,4,1,1)
        self.layout.addWidget(QLabel(""), 0, 5, 1, 3)
        self.layout.addWidget(self.btnCreateGraph, 0, 8, 1, 1)
        #self.layout.addWidget(QLabel("Choose Data Filter:"), 0, 6, 1, 1)
        #self.layout.addWidget(self.filter_data,0,7,1,2)
        self.layout.addWidget(self.groupBox1,1,0,5,5)
        self.layout.addWidget(QLabel(""), 1, 5, 5, 1)
        self.layout.addWidget(self.groupBox2, 1, 6, 5, 3)

        self.setCentralWidget(self.main_widget)
        self.resize(1400, 700)
        self.show()
        self.updateCategory()

    def updateCategory(self):
        self.dropdown2.clear()
        feature_category = self.dropdown1.currentText()
        if(feature_category=="Personal"):
            self.featuresList=personal_features
        elif (feature_category == "Organisation"):
            self.featuresList = organisation_features
        elif (feature_category == "Commution"):
            self.featuresList = commution_features
        elif (feature_category == "Satisfaction"):
            self.featuresList = satisfaction_features
        self.dropdown2.addItems(self.featuresList)



    def update(self):
        #::--------------------------------------------------------
        # This method executes each time a change is made on the canvas
        # containing the elements of the graph
        # The purpose of the method es to draw a dot graph using the
        # score of happiness and the feature chosen the canvas
        #::--------------------------------------------------------
        colors=["b", "r", "g", "y", "k", "c"]
        self.ax1.clear()
        self.ax2.clear()
        self.filtered_data = attr_data.copy()
        graph_feature1 = self.dropdown2.currentText()
        val1 = []
        val2 = []
        if (graph_feature1 in categorical_features):
            self.filtered_data["Count"]=1
            category_values=self.filtered_data[graph_feature1].unique()

            yes_data=self.filtered_data[self.filtered_data["Attrition"]=="Yes"]
            no_data=self.filtered_data[self.filtered_data["Attrition"]=="No"]

            my_pt_yes = pd.pivot_table(yes_data, index=graph_feature1, values="Count",aggfunc=np.sum)
            my_pt_yes = pd.DataFrame(my_pt_yes.to_records())
            my_dict_yes=dict(zip(my_pt_yes[graph_feature1], my_pt_yes["Count"]))

            my_pt_no = pd.pivot_table(no_data, index=graph_feature1, values="Count", aggfunc=np.sum)
            my_pt_no = pd.DataFrame(my_pt_no.to_records())
            my_dict_no = dict(zip(my_pt_no[graph_feature1], my_pt_no["Count"]))
            for temp_value in category_values:
                if temp_value in my_dict_yes.keys():
                    val1.append(-1*(my_dict_yes[temp_value]))
                else:
                    val1.append(0)

                if temp_value in my_dict_no.keys():
                    val2.append(my_dict_no[temp_value])
                else:
                    val2.append(0)

        else:
            yes_data = self.filtered_data[self.filtered_data["Attrition"] == "Yes"]
            no_data = self.filtered_data[self.filtered_data["Attrition"] == "No"]

            category_values=["Max","Median","Mean","Min"]
            val1.append(-1 * (yes_data[graph_feature1].max()))
            val1.append(-1 * (yes_data[graph_feature1].median(skipna=True)))
            val1.append(-1 * (yes_data[graph_feature1].mean(skipna=True)))
            val1.append(-1*(yes_data[graph_feature1].min()))

            val2.append(no_data[graph_feature1].max())
            val2.append(no_data[graph_feature1].median(skipna=True))
            val2.append(no_data[graph_feature1].mean(skipna=True))
            val2.append(no_data[graph_feature1].min())


        self.ax1.barh(category_values, val1, color='blue')
        self.ax1.set_title("Attrition: No")
        self.ax1.axis('off')
        #self.ax1.set_xlabel(cat1)
        #self.ax1.set_ylabel("Count")
        self.ax1.grid(False)

        self.ax2.barh(category_values, val2, color='red')
        self.ax2.set_title("Attrition: Yes")
        self.ax2.axis('off')
        #self.ax2.set_xlabel(cat1)
        #self.ax2.set_ylabel("Count")
        self.ax2.grid(False)

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()


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
        self.feature20 = QCheckBox(features_list[20], self)
        self.feature21 = QCheckBox(features_list[21], self)
        self.feature22 = QCheckBox(features_list[22], self)
        self.feature23 = QCheckBox(features_list[23], self)
        self.feature24 = QCheckBox(features_list[24], self)
        self.feature25 = QCheckBox(features_list[25], self)
        self.feature26 = QCheckBox(features_list[26], self)
        self.feature27 = QCheckBox(features_list[27], self)
        self.feature28 = QCheckBox(features_list[28], self)
        self.feature29 = QCheckBox(features_list[29], self)
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
        self.feature20.setChecked(True)
        self.feature21.setChecked(True)
        self.feature22.setChecked(True)
        self.feature23.setChecked(True)
        self.feature24.setChecked(True)
        self.feature25.setChecked(True)
        self.feature26.setChecked(True)
        self.feature27.setChecked(True)
        self.feature28.setChecked(True)
        self.feature29.setChecked(True)

        self.lblPercentTest = QLabel('Percentage for Test :')
        self.lblPercentTest.adjustSize()

        self.txtPercentTest = QLineEdit(self)
        self.txtPercentTest.setText("20")

        self.lblEstimatorCount = QLabel('Neighbours:')
        self.lblEstimatorCount.adjustSize()

        self.txtEstimatorCount = QLineEdit(self)
        self.txtEstimatorCount.setText("35")

        self.btnExecute = QPushButton("Run Model")
        self.btnExecute.clicked.connect(self.update)

        self.groupBox1Layout.addWidget(self.feature0,0,0,1,1)
        self.groupBox1Layout.addWidget(self.feature1,0,1,1,1)
        self.groupBox1Layout.addWidget(self.feature2,1,0,1,1)
        self.groupBox1Layout.addWidget(self.feature3,1,1,1,1)
        self.groupBox1Layout.addWidget(self.feature4,2,0,1,1)
        self.groupBox1Layout.addWidget(self.feature5,2,1,1,1)
        self.groupBox1Layout.addWidget(self.feature6,3,0,1,1)
        self.groupBox1Layout.addWidget(self.feature7,3,1,1,1)
        self.groupBox1Layout.addWidget(self.feature8, 4, 0,1,1)
        self.groupBox1Layout.addWidget(self.feature9, 4, 1,1,1)
        self.groupBox1Layout.addWidget(self.feature10, 5, 0,1,1)
        self.groupBox1Layout.addWidget(self.feature11, 5, 1,1,1)
        self.groupBox1Layout.addWidget(self.feature12, 6, 0,1,1)
        self.groupBox1Layout.addWidget(self.feature13, 6, 1,1,1)
        self.groupBox1Layout.addWidget(self.feature14, 7, 0,1,1)
        self.groupBox1Layout.addWidget(self.feature15, 7, 1,1,1)
        self.groupBox1Layout.addWidget(self.feature16, 8, 0,1,1)
        self.groupBox1Layout.addWidget(self.feature17, 8, 1,1,1)
        self.groupBox1Layout.addWidget(self.feature18, 9, 0,1,1)
        self.groupBox1Layout.addWidget(self.feature19, 9, 1,1,1)
        self.groupBox1Layout.addWidget(self.feature20, 10, 0,1,1)
        self.groupBox1Layout.addWidget(self.feature21, 10, 1,1,1)
        self.groupBox1Layout.addWidget(self.feature22, 11, 0,1,1)
        self.groupBox1Layout.addWidget(self.feature23, 11, 1,1,1)
        self.groupBox1Layout.addWidget(self.feature24, 12, 0,1,1)
        self.groupBox1Layout.addWidget(self.feature25, 12, 1,1,1)
        self.groupBox1Layout.addWidget(self.feature26, 13, 0,1,1)
        self.groupBox1Layout.addWidget(self.feature27, 13, 1,1,1)
        self.groupBox1Layout.addWidget(self.feature28, 14, 0,1,1)
        self.groupBox1Layout.addWidget(self.feature29, 14, 1,1,1)
        self.groupBox1Layout.addWidget(self.lblPercentTest, 15, 0,1,1)
        self.groupBox1Layout.addWidget(self.txtPercentTest, 15, 1,1,1)
        self.groupBox1Layout.addWidget(self.lblEstimatorCount, 16, 0, 1, 1)
        self.groupBox1Layout.addWidget(self.txtEstimatorCount, 16, 1, 1, 1)
        self.groupBox1Layout.addWidget(self.btnExecute, 17, 0,1,2)

        self.groupBox2 = QGroupBox('Results from the model')
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)
        self.groupBox2.setMinimumSize(400, 50)

        self.lblResults = QLabel('Results:')
        self.lblResults.adjustSize()
        self.txtResults = QPlainTextEdit()
        #self.txtResults.setMinimumSize(200,100)
        self.lblAccuracy = QLabel('Accuracy:')
        self.txtAccuracy = QLineEdit()

        self.groupBox2Layout.addWidget(self.lblResults)
        self.groupBox2Layout.addWidget(self.txtResults)
        self.groupBox2Layout.addWidget(self.lblAccuracy)
        self.groupBox2Layout.addWidget(self.txtAccuracy)


        self.groupBox3 = QGroupBox('Summary and Comparison')
        self.groupBox3Layout = QVBoxLayout()
        self.groupBox3.setLayout(self.groupBox3Layout)
        self.groupBox3.setMinimumSize(400, 50)

        self.lbl_summary = QLabel('Summary:')
        self.lbl_summary.adjustSize()
        self.txt_summary = QPlainTextEdit()
        self.lbl_other_models = QLabel('Other Models Performance:')
        self.other_models = QWidget(self)
        self.other_models.layout = QFormLayout(self.other_models)
        #self.other_modelsLayout = QFormLayout()
        #self.other_models.setLayout(self.other_modelsLayout)
        self.txtAccuracy_lr = QLineEdit()
        self.txtAccuracy_knn = QLineEdit()
        self.txtAccuracy_dt = QLineEdit()
        self.other_models.layout.addRow('Logistic:', self.txtAccuracy_lr)
        self.other_models.layout.addRow('KNN:', self.txtAccuracy_knn)
        self.other_models.layout.addRow('Decision Trees:', self.txtAccuracy_dt)

        self.groupBox3Layout.addWidget(self.lbl_summary)
        self.groupBox3Layout.addWidget(self.txt_summary)
        self.groupBox3Layout.addWidget(self.lbl_other_models)
        self.groupBox3Layout.addWidget(self.other_models)


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

        self.groupBoxG2 = QGroupBox('AUC Score Vs Number of Trees')
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

        self.layout.addWidget(self.groupBox1,0,0,2,1)
        self.layout.addWidget(self.groupBoxG1,0,1,1,1)
        self.layout.addWidget(self.groupBoxG3,0,2,1,1)
        self.layout.addWidget(self.groupBox2,0,3,1,1)
        self.layout.addWidget(self.groupBoxG2,1,1,1,1)
        self.layout.addWidget(self.groupBoxG4,1,2,1,1)
        self.layout.addWidget(self.groupBox3,1,3,1,1)

        self.setCentralWidget(self.main_widget)
        self.resize(1800, 900)
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
        estimator_input = round(float(self.txtEstimatorCount.text()))

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
        self.clf_rf = RandomForestClassifier(n_estimators=estimator_input, random_state=500)

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
                self.ax1.text(j, i, str(conf_matrix[i][j]))

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

        ## End Graph1 -- Confusion Matrix

        #::----------------------------------------
        ## Graph 2 - ROC Curve
        #::----------------------------------------

        auc_test = []
        auc_train = []
        estimator_count= [1, 2, 4, 8, 16, 32, 50, 64, 100, 200, 300]
        # Might take some time
        for i in estimator_count:
            self.rf_graph = RandomForestClassifier(n_estimators=i)
            self.rf_graph.fit(X_train, y_train)
            temp_train_pred=self.rf_graph.predict(X_train)
            temp_test_pred=self.rf_graph.predict(X_test)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, temp_train_pred)
            auc_train.append(auc(false_positive_rate, true_positive_rate))
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, temp_test_pred)
            auc_test.append(auc(false_positive_rate, true_positive_rate))

        self.ax2.plot(estimator_count,auc_train , color='blue', label="Train AUC")
        self.ax2.plot(estimator_count, auc_test, color='red', label="Test AUC")
        self.ax2.set_xlabel('Number of Trees')
        self.ax2.set_ylabel('AUC Score')
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
        y_test_bin = pd.get_dummies(y_test).to_numpy()
        n_classes = y_test_bin.shape[1]

        # From the sckict learn site
        # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        # print(pd.get_dummies(y_test).to_numpy().ravel())

        # print("\n\n********************************\n\n")
        # print(y_pred_score.ravel())
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_pred_score.ravel())

        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        lw = 2
        str_classes= ['No','Yes']
        colors = cycle(['magenta', 'darkorange'])
        for i, color in zip(range(n_classes), colors):
            self.ax4.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='{0} (area = {1:0.2f})'
                           ''.format(str_classes[i], roc_auc[i]))

        self.ax4.plot([0, 1], [0, 1], 'k--', lw=lw)
        self.ax4.set_xlim([0.0, 1.0])
        self.ax4.set_ylim([0.0, 1.05])
        self.ax4.set_xlabel('False Positive Rate')
        self.ax4.set_ylabel('True Positive Rate')
        self.ax4.legend(loc="lower right")

        # show the plot
        self.fig4.tight_layout()
        self.fig4.canvas.draw_idle()

        #::-----------------------------
        # End of graph 4  - ROC curve by class
        #::-----------------------------

        #::-----------------------------------------------------
        # Other Models Comparison
        #::-----------------------------------------------------

        self.other_clf_lr=LogisticRegression(random_state=500)
        self.other_clf_lr.fit(X_train, y_train)
        y_pred_lr = self.other_clf_lr.predict(X_test)
        self.accuracy_lr=accuracy_score(y_test, y_pred_lr) *100
        self.txtAccuracy_lr.setText(str(self.accuracy_lr))

        self.other_clf_dt = DecisionTreeClassifier(criterion="gini")
        self.other_clf_dt.fit(X_train, y_train)
        y_pred_dt = self.other_clf_dt.predict(X_test)
        self.accuracy_dt = accuracy_score(y_test, y_pred_dt) * 100
        self.txtAccuracy_dt.setText(str(self.accuracy_dt))

        self.other_clf_knn = KNeighborsClassifier(n_neighbors=9)
        self.other_clf_knn.fit(X_train, y_train)
        y_pred_knn = self.other_clf_knn.predict(X_test)
        self.accuracy_knn = accuracy_score(y_test, y_pred_knn) * 100
        self.txtAccuracy_knn.setText(str(self.accuracy_knn))

        #::-----------------------------
        # End of Other Models Comparison
        #::-----------------------------

class DecisionTree(QMainWindow):
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
        super(DecisionTree, self).__init__()
        self.Title = "Decision Tree Classifier"
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

        self.groupBox1 = QGroupBox('Decision Tree Features')
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
        self.feature20 = QCheckBox(features_list[20], self)
        self.feature21 = QCheckBox(features_list[21], self)
        self.feature22 = QCheckBox(features_list[22], self)
        self.feature23 = QCheckBox(features_list[23], self)
        self.feature24 = QCheckBox(features_list[24], self)
        self.feature25 = QCheckBox(features_list[25], self)
        self.feature26 = QCheckBox(features_list[26], self)
        self.feature27 = QCheckBox(features_list[27], self)
        self.feature28 = QCheckBox(features_list[28], self)
        self.feature29 = QCheckBox(features_list[29], self)
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
        self.feature20.setChecked(True)
        self.feature21.setChecked(True)
        self.feature22.setChecked(True)
        self.feature23.setChecked(True)
        self.feature24.setChecked(True)
        self.feature25.setChecked(True)
        self.feature26.setChecked(True)
        self.feature27.setChecked(True)
        self.feature28.setChecked(True)
        self.feature29.setChecked(True)

        self.lblPercentTest = QLabel('Percentage for Test :')
        self.lblPercentTest.adjustSize()

        self.txtPercentTest = QLineEdit(self)
        self.txtPercentTest.setText("20")

        self.btnExecute = QPushButton("Run Model")
        self.btnExecute.clicked.connect(self.update)

        self.groupBox1Layout.addWidget(self.feature0,0,0,1,1)
        self.groupBox1Layout.addWidget(self.feature1,0,1,1,1)
        self.groupBox1Layout.addWidget(self.feature2,1,0,1,1)
        self.groupBox1Layout.addWidget(self.feature3,1,1,1,1)
        self.groupBox1Layout.addWidget(self.feature4,2,0,1,1)
        self.groupBox1Layout.addWidget(self.feature5,2,1,1,1)
        self.groupBox1Layout.addWidget(self.feature6,3,0,1,1)
        self.groupBox1Layout.addWidget(self.feature7,3,1,1,1)
        self.groupBox1Layout.addWidget(self.feature8, 4, 0,1,1)
        self.groupBox1Layout.addWidget(self.feature9, 4, 1,1,1)
        self.groupBox1Layout.addWidget(self.feature10, 5, 0,1,1)
        self.groupBox1Layout.addWidget(self.feature11, 5, 1,1,1)
        self.groupBox1Layout.addWidget(self.feature12, 6, 0,1,1)
        self.groupBox1Layout.addWidget(self.feature13, 6, 1,1,1)
        self.groupBox1Layout.addWidget(self.feature14, 7, 0,1,1)
        self.groupBox1Layout.addWidget(self.feature15, 7, 1,1,1)
        self.groupBox1Layout.addWidget(self.feature16, 8, 0,1,1)
        self.groupBox1Layout.addWidget(self.feature17, 8, 1,1,1)
        self.groupBox1Layout.addWidget(self.feature18, 9, 0,1,1)
        self.groupBox1Layout.addWidget(self.feature19, 9, 1,1,1)
        self.groupBox1Layout.addWidget(self.feature20, 10, 0,1,1)
        self.groupBox1Layout.addWidget(self.feature21, 10, 1,1,1)
        self.groupBox1Layout.addWidget(self.feature22, 11, 0,1,1)
        self.groupBox1Layout.addWidget(self.feature23, 11, 1,1,1)
        self.groupBox1Layout.addWidget(self.feature24, 12, 0,1,1)
        self.groupBox1Layout.addWidget(self.feature25, 12, 1,1,1)
        self.groupBox1Layout.addWidget(self.feature26, 13, 0,1,1)
        self.groupBox1Layout.addWidget(self.feature27, 13, 1,1,1)
        self.groupBox1Layout.addWidget(self.feature28, 14, 0,1,1)
        self.groupBox1Layout.addWidget(self.feature29, 14, 1,1,1)
        self.groupBox1Layout.addWidget(self.lblPercentTest, 15, 0,1,1)
        self.groupBox1Layout.addWidget(self.txtPercentTest, 15, 1,1,1)
        self.groupBox1Layout.addWidget(self.btnExecute, 16, 0,1,2)

        self.groupBox2 = QGroupBox('Results from the model')
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)
        self.groupBox2.setMinimumSize(400, 50)

        self.lblResults = QLabel('Results:')
        self.lblResults.adjustSize()
        self.txtResults = QPlainTextEdit()
        #self.txtResults.setMinimumSize(200,100)
        self.lblAccuracy = QLabel('Accuracy:')
        self.txtAccuracy = QLineEdit()

        self.groupBox2Layout.addWidget(self.lblResults)
        self.groupBox2Layout.addWidget(self.txtResults)
        self.groupBox2Layout.addWidget(self.lblAccuracy)
        self.groupBox2Layout.addWidget(self.txtAccuracy)


        self.groupBox3 = QGroupBox('Summary and Comparison')
        self.groupBox3Layout = QVBoxLayout()
        self.groupBox3.setLayout(self.groupBox3Layout)
        self.groupBox3.setMinimumSize(400, 50)

        self.lbl_summary = QLabel('Summary:')
        self.lbl_summary.adjustSize()
        self.txt_summary = QPlainTextEdit()
        self.lbl_other_models = QLabel('Other Models Performance:')
        self.other_models = QWidget(self)
        self.other_models.layout = QFormLayout(self.other_models)
        #self.other_modelsLayout = QFormLayout()
        #self.other_models.setLayout(self.other_modelsLayout)
        self.txtAccuracy_lr = QLineEdit()
        self.txtAccuracy_knn = QLineEdit()
        self.txtAccuracy_rf = QLineEdit()
        self.other_models.layout.addRow('Logistic:', self.txtAccuracy_lr)
        self.other_models.layout.addRow('KNN:', self.txtAccuracy_knn)
        self.other_models.layout.addRow('Random Forest:', self.txtAccuracy_rf)

        self.groupBox3Layout.addWidget(self.lbl_summary)
        self.groupBox3Layout.addWidget(self.txt_summary)
        self.groupBox3Layout.addWidget(self.lbl_other_models)
        self.groupBox3Layout.addWidget(self.other_models)


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

        self.labelImage = QLabel(self)
        self.image = QPixmap()
        self.labelImage.setPixmap(self.image)
        self.image_area = QScrollArea()
        self.image_area.setWidget(self.labelImage)
        self.labelImage.setPixmap(QPixmap("temp_background.png"))
        self.labelImage.adjustSize()
        #vbox.addWidget(labelImage)

        #self.image = QPixmap()
        #label.setPixmap(pixmap)
        #self.resize(pixmap.width(), pixmap.height())
        #self.canvas2 = FigureCanvas(self.image)

        #self.canvas2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        #self.canvas2.updateGeometry()

        self.groupBoxG2 = QGroupBox('Decision Tree Graph')
        self.groupBoxG2Layout = QVBoxLayout()
        self.groupBoxG2.setLayout(self.groupBoxG2Layout)
        self.groupBoxG2Layout.addWidget(self.image_area)

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

        self.layout.addWidget(self.groupBox1,0,0,2,1)
        self.layout.addWidget(self.groupBoxG1,0,1,1,1)
        self.layout.addWidget(self.groupBoxG3,0,2,1,1)
        self.layout.addWidget(self.groupBox2,0,3,1,1)
        self.layout.addWidget(self.groupBoxG2,1,1,1,1)
        self.layout.addWidget(self.groupBoxG4,1,2,1,1)
        self.layout.addWidget(self.groupBox3,1,3,1,1)

        self.setCentralWidget(self.main_widget)
        self.resize(1800, 900)
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
        self.clf_dt =DecisionTreeClassifier(criterion="gini")

        # perform training
        self.clf_dt.fit(X_train, y_train)

        #-----------------------------------------------------------------------

        # predicton on test using all features
        y_pred = self.clf_dt.predict(X_test)
        y_pred_score = self.clf_dt.predict_proba(X_test)


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
                self.ax1.text(j, i, str(conf_matrix[i][j]))

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

        ## End Graph1 -- Confusion Matrix

        #::----------------------------------------
        ## Graph 2 - ROC Curve
        #::----------------------------------------



        #self.fig2.tight_layout()
        #self.fig2.canvas.draw_idle()

        dot_data = export_graphviz(self.clf_dt,
                                        feature_names=X_train.columns,
                                        class_names=['No', 'Yes'],
                                        out_file=None,
                                        filled=True,
                                        rounded=True,
                                        max_depth=3)

        graph = graph_from_dot_data(dot_data)
        colors = ('turquoise', 'orange')
        edges = collections.defaultdict(list)

        for edge in graph.get_edge_list():
            edges[edge.get_source()].append(int(edge.get_destination()))

        for edge in edges:
            edges[edge].sort()
            for i in range(2):
                dest = graph.get_node(str(edges[edge][i]))[0]
                dest.set_fillcolor(colors[i])
        graph.set_size('"15,15!"')
        graph.write_png('DecisionTree_Attrition.png')
        self.labelImage.setPixmap(QPixmap("DecisionTree_Attrition.png"))
        self.labelImage.adjustSize()

        ######################################
        # Graph - 3 Feature Importances
        #####################################
        # get feature importances

        importances = self.clf_dt.feature_importances_

        # convert the importances into one-dimensional 1darray with corresponding df column names as axis labels
        f_importances = pd.Series(importances, X_dt.columns)
        #print(f_importances)
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
        # y_test_bin = label_binarize(y_test, classes=[0, 1])
        # print(pd.get_dummies(y_test))
        # print(pd.get_dummies(y_test).to_numpy())
        y_test_bin = pd.get_dummies(y_test).to_numpy()
        n_classes = y_test_bin.shape[1]

        # From the sckict learn site
        # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        # print(pd.get_dummies(y_test).to_numpy().ravel())

        # print("\n\n********************************\n\n")
        # print(y_pred_score.ravel())
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_pred_score.ravel())

        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        lw = 2

        str_classes= ['No','Yes']
        colors = cycle(['magenta', 'darkorange'])
        for i, color in zip(range(n_classes), colors):
            self.ax4.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='{0} (area = {1:0.2f})'
                           ''.format(str_classes[i], roc_auc[i]))

        self.ax4.plot([0, 1], [0, 1], 'k--', lw=lw)
        self.ax4.set_xlim([0.0, 1.0])
        self.ax4.set_ylim([0.0, 1.05])
        self.ax4.set_xlabel('False Positive Rate')
        self.ax4.set_ylabel('True Positive Rate')
        self.ax4.legend(loc="lower right")

        # show the plot
        self.fig4.tight_layout()
        self.fig4.canvas.draw_idle()

        #::-----------------------------
        # End of graph 4  - ROC curve by class
        #::-----------------------------

        #::-----------------------------------------------------
        # Other Models Comparison
        #::-----------------------------------------------------

        self.other_clf_lr=LogisticRegression(random_state=500)
        self.other_clf_lr.fit(X_train, y_train)
        y_pred_lr = self.other_clf_lr.predict(X_test)
        self.accuracy_lr=accuracy_score(y_test, y_pred_lr) *100
        self.txtAccuracy_lr.setText(str(self.accuracy_lr))

        self.other_clf_rf = RandomForestClassifier(n_estimators=100, random_state=500)
        self.other_clf_rf.fit(X_train, y_train)
        y_pred_rf = self.other_clf_rf.predict(X_test)
        self.accuracy_rf = accuracy_score(y_test, y_pred_rf) * 100
        self.txtAccuracy_rf.setText(str(self.accuracy_rf))

        self.other_clf_knn = KNeighborsClassifier(n_neighbors=9)
        self.other_clf_knn.fit(X_train, y_train)
        y_pred_knn = self.other_clf_knn.predict(X_test)
        self.accuracy_knn = accuracy_score(y_test, y_pred_knn) * 100
        self.txtAccuracy_knn.setText(str(self.accuracy_knn))

        #::-----------------------------
        # End of Other Models Comparison
        #::-----------------------------

class LogisticRegressionClassifier(QMainWindow):
    #::--------------------------------------------------------------------------------
    # Implementation of LOgistic Regression using the happiness dataset
    # the methods in this class are
    #       _init_ : initialize the class
    #       initUi : creates the canvas and all the elements in the canvas
    #       update : populates the elements of the canvas base on the parametes
    #               chosen by the user
    #::---------------------------------------------------------------------------------
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(LogisticRegressionClassifier, self).__init__()
        self.Title = "Logistic Regression"
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

        self.groupBox1 = QGroupBox('Logistic Regression Features')
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
        self.feature20 = QCheckBox(features_list[20], self)
        self.feature21 = QCheckBox(features_list[21], self)
        self.feature22 = QCheckBox(features_list[22], self)
        self.feature23 = QCheckBox(features_list[23], self)
        self.feature24 = QCheckBox(features_list[24], self)
        self.feature25 = QCheckBox(features_list[25], self)
        self.feature26 = QCheckBox(features_list[26], self)
        self.feature27 = QCheckBox(features_list[27], self)
        self.feature28 = QCheckBox(features_list[28], self)
        self.feature29 = QCheckBox(features_list[29], self)
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
        self.feature20.setChecked(True)
        self.feature21.setChecked(True)
        self.feature22.setChecked(True)
        self.feature23.setChecked(True)
        self.feature24.setChecked(True)
        self.feature25.setChecked(True)
        self.feature26.setChecked(True)
        self.feature27.setChecked(True)
        self.feature28.setChecked(True)
        self.feature29.setChecked(True)

        self.lblPercentTest = QLabel('Percentage for Test :')
        self.lblPercentTest.adjustSize()

        self.txtPercentTest = QLineEdit(self)
        self.txtPercentTest.setText("20")

        self.btnExecute = QPushButton("Run Model")
        self.btnExecute.clicked.connect(self.update)

        self.groupBox1Layout.addWidget(self.feature0,0,0,1,1)
        self.groupBox1Layout.addWidget(self.feature1,0,1,1,1)
        self.groupBox1Layout.addWidget(self.feature2,1,0,1,1)
        self.groupBox1Layout.addWidget(self.feature3,1,1,1,1)
        self.groupBox1Layout.addWidget(self.feature4,2,0,1,1)
        self.groupBox1Layout.addWidget(self.feature5,2,1,1,1)
        self.groupBox1Layout.addWidget(self.feature6,3,0,1,1)
        self.groupBox1Layout.addWidget(self.feature7,3,1,1,1)
        self.groupBox1Layout.addWidget(self.feature8, 4, 0,1,1)
        self.groupBox1Layout.addWidget(self.feature9, 4, 1,1,1)
        self.groupBox1Layout.addWidget(self.feature10, 5, 0,1,1)
        self.groupBox1Layout.addWidget(self.feature11, 5, 1,1,1)
        self.groupBox1Layout.addWidget(self.feature12, 6, 0,1,1)
        self.groupBox1Layout.addWidget(self.feature13, 6, 1,1,1)
        self.groupBox1Layout.addWidget(self.feature14, 7, 0,1,1)
        self.groupBox1Layout.addWidget(self.feature15, 7, 1,1,1)
        self.groupBox1Layout.addWidget(self.feature16, 8, 0,1,1)
        self.groupBox1Layout.addWidget(self.feature17, 8, 1,1,1)
        self.groupBox1Layout.addWidget(self.feature18, 9, 0,1,1)
        self.groupBox1Layout.addWidget(self.feature19, 9, 1,1,1)
        self.groupBox1Layout.addWidget(self.feature20, 10, 0,1,1)
        self.groupBox1Layout.addWidget(self.feature21, 10, 1,1,1)
        self.groupBox1Layout.addWidget(self.feature22, 11, 0,1,1)
        self.groupBox1Layout.addWidget(self.feature23, 11, 1,1,1)
        self.groupBox1Layout.addWidget(self.feature24, 12, 0,1,1)
        self.groupBox1Layout.addWidget(self.feature25, 12, 1,1,1)
        self.groupBox1Layout.addWidget(self.feature26, 13, 0,1,1)
        self.groupBox1Layout.addWidget(self.feature27, 13, 1,1,1)
        self.groupBox1Layout.addWidget(self.feature28, 14, 0,1,1)
        self.groupBox1Layout.addWidget(self.feature29, 14, 1,1,1)
        self.groupBox1Layout.addWidget(self.lblPercentTest, 15, 0,1,1)
        self.groupBox1Layout.addWidget(self.txtPercentTest, 15, 1,1,1)
        self.groupBox1Layout.addWidget(self.btnExecute, 16, 0,1,2)

        self.groupBox2 = QGroupBox('Results from the model')
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)
        self.groupBox2.setMinimumSize(400, 50)

        self.lblResults = QLabel('Results:')
        self.lblResults.adjustSize()
        self.txtResults = QPlainTextEdit()
        #self.txtResults.setMinimumSize(200,100)
        self.lblAccuracy = QLabel('Accuracy:')
        self.txtAccuracy = QLineEdit()

        self.groupBox2Layout.addWidget(self.lblResults)
        self.groupBox2Layout.addWidget(self.txtResults)
        self.groupBox2Layout.addWidget(self.lblAccuracy)
        self.groupBox2Layout.addWidget(self.txtAccuracy)


        self.groupBox3 = QGroupBox('Summary and Comparison')
        self.groupBox3Layout = QVBoxLayout()
        self.groupBox3.setLayout(self.groupBox3Layout)
        self.groupBox3.setMinimumSize(400, 50)

        self.lbl_summary = QLabel('Summary:')
        self.lbl_summary.adjustSize()
        self.txt_summary = QPlainTextEdit()
        self.lbl_other_models = QLabel('Other Models Performance:')
        self.other_models = QWidget(self)
        self.other_models.layout = QFormLayout(self.other_models)
        #self.other_modelsLayout = QFormLayout()
        #self.other_models.setLayout(self.other_modelsLayout)
        self.txtAccuracy_rf = QLineEdit()
        self.txtAccuracy_knn = QLineEdit()
        self.txtAccuracy_dt = QLineEdit()
        self.other_models.layout.addRow('Random Forest:', self.txtAccuracy_rf)
        self.other_models.layout.addRow('KNN:', self.txtAccuracy_knn)
        self.other_models.layout.addRow('Decision Trees:', self.txtAccuracy_dt)

        self.groupBox3Layout.addWidget(self.lbl_summary)
        self.groupBox3Layout.addWidget(self.txt_summary)
        self.groupBox3Layout.addWidget(self.lbl_other_models)
        self.groupBox3Layout.addWidget(self.other_models)


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

        self.groupBoxG2 = QGroupBox('Calibration Curve')
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

        self.groupBoxG3 = QGroupBox('Cross Validation Score')
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

        self.layout.addWidget(self.groupBox1,0,0,2,1)
        self.layout.addWidget(self.groupBoxG1,0,1,1,1)
        self.layout.addWidget(self.groupBoxG3,0,2,1,1)
        self.layout.addWidget(self.groupBox2,0,3,1,1)
        self.layout.addWidget(self.groupBoxG2,1,1,1,1)
        self.layout.addWidget(self.groupBoxG4,1,2,1,1)
        self.layout.addWidget(self.groupBox3,1,3,1,1)

        self.setCentralWidget(self.main_widget)
        self.resize(1800, 900)
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
        self.clf_lr = LogisticRegression()

        # perform training
        self.clf_lr.fit(X_train, y_train)

        #-----------------------------------------------------------------------

        # predicton on test using all features
        y_pred = self.clf_lr.predict(X_test)
        y_pred_score = self.clf_lr.predict_proba(X_test)

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
                self.ax1.text(j, i, str(conf_matrix[i][j]))

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

        ## End Graph1 -- Confusion Matrix

        #::----------------------------------------
        ## Graph 2 - ROC Curve
        #::----------------------------------------

        logreg_y, logreg_x = calibration_curve(y_test, y_pred_score[:,1], n_bins=10)
        self.ax2.plot(logreg_x, logreg_y, marker='o', linewidth=1)
        self.ax2.plot(np.linspace(0,1,10), np.linspace(0,1,10), linewidth=1, color="black")
        self.ax2.set_xlabel('Predicted probability')
        self.ax2.set_ylabel('True probability in each bin')

        # show the plot
        self.fig2.tight_layout()
        self.fig2.canvas.draw_idle()


        ######################################
        # Graph - 3 Feature Importances
        #####################################
        # get feature importances
        scores_arr = []
        #feature_arr = []
        for val in (X_train.columns):
            #print(val)
            cvs_X = X_train[val].values.reshape(-1, 1)
            #print(cvs_X)
            scores = cross_val_score(self.clf_lr, cvs_X, y_train, cv=10)
            scores_arr.append(scores.mean())

        #importances = self.clf_knn.feature_importances_

        # convert the importances into one-dimensional 1darray with corresponding df column names as axis labels
        f_importances = pd.Series(scores_arr, X_train.columns)
        # sort the array in descending order of the importances
        f_importances.sort_values(ascending=False, inplace=True)
        f_importances=f_importances[0:20]
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
        # y_test_bin = label_binarize(y_test, classes=[0, 1])
        # print(pd.get_dummies(y_test))
        # print(pd.get_dummies(y_test).to_numpy())
        y_test_bin = pd.get_dummies(y_test).to_numpy()
        n_classes = y_test_bin.shape[1]

        # From the sckict learn site
        # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        # print(pd.get_dummies(y_test).to_numpy().ravel())

        # print("\n\n********************************\n\n")
        # print(y_pred_score.ravel())
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_pred_score.ravel())

        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        lw = 2
        str_classes= ['No','Yes']
        colors = cycle(['magenta', 'darkorange'])
        for i, color in zip(range(n_classes), colors):
            self.ax4.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='{0} (area = {1:0.2f})'
                           ''.format(str_classes[i], roc_auc[i]))

        self.ax4.plot([0, 1], [0, 1], 'k--', lw=lw)
        self.ax4.set_xlim([0.0, 1.0])
        self.ax4.set_ylim([0.0, 1.05])
        self.ax4.set_xlabel('False Positive Rate')
        self.ax4.set_ylabel('True Positive Rate')
        self.ax4.legend(loc="lower right")

        # show the plot
        self.fig4.tight_layout()
        self.fig4.canvas.draw_idle()

        #::-----------------------------
        # End of graph 4  - ROC curve by class
        #::-----------------------------

        #::-----------------------------------------------------
        # Other Models Comparison
        #::-----------------------------------------------------

        self.other_clf_rf=RandomForestClassifier(n_estimators=100, random_state=500)
        self.other_clf_rf.fit(X_train, y_train)
        y_pred_rf = self.other_clf_rf.predict(X_test)
        self.accuracy_rf=accuracy_score(y_test, y_pred_rf) *100
        self.txtAccuracy_rf.setText(str(self.accuracy_rf))

        self.other_clf_dt = DecisionTreeClassifier(criterion="gini")
        self.other_clf_dt.fit(X_train, y_train)
        y_pred_dt = self.other_clf_dt.predict(X_test)
        self.accuracy_dt = accuracy_score(y_test, y_pred_dt) * 100
        self.txtAccuracy_dt.setText(str(self.accuracy_dt))

        self.other_clf_knn = KNeighborsClassifier(n_neighbors=9)
        self.other_clf_knn.fit(X_train, y_train)
        y_pred_knn = self.other_clf_knn.predict(X_test)
        self.accuracy_knn = accuracy_score(y_test, y_pred_knn) * 100
        self.txtAccuracy_knn.setText(str(self.accuracy_knn))

        #::-----------------------------
        # End of Other Models Comparison
        #::-----------------------------


class KNNClassifier(QMainWindow):
    #::--------------------------------------------------------------------------------
    # Implementation of LOgistic Regression using the happiness dataset
    # the methods in this class are
    #       _init_ : initialize the class
    #       initUi : creates the canvas and all the elements in the canvas
    #       update : populates the elements of the canvas base on the parametes
    #               chosen by the user
    #::---------------------------------------------------------------------------------
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(KNNClassifier, self).__init__()
        self.Title = "K-Nearst Neighbor"
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

        self.groupBox1 = QGroupBox('K - Nearest Neighbor Features')
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
        self.feature20 = QCheckBox(features_list[20], self)
        self.feature21 = QCheckBox(features_list[21], self)
        self.feature22 = QCheckBox(features_list[22], self)
        self.feature23 = QCheckBox(features_list[23], self)
        self.feature24 = QCheckBox(features_list[24], self)
        self.feature25 = QCheckBox(features_list[25], self)
        self.feature26 = QCheckBox(features_list[26], self)
        self.feature27 = QCheckBox(features_list[27], self)
        self.feature28 = QCheckBox(features_list[28], self)
        self.feature29 = QCheckBox(features_list[29], self)
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
        self.feature20.setChecked(True)
        self.feature21.setChecked(True)
        self.feature22.setChecked(True)
        self.feature23.setChecked(True)
        self.feature24.setChecked(True)
        self.feature25.setChecked(True)
        self.feature26.setChecked(True)
        self.feature27.setChecked(True)
        self.feature28.setChecked(True)
        self.feature29.setChecked(True)

        self.lblPercentTest = QLabel('Percentage for Test :')
        self.lblPercentTest.adjustSize()

        self.txtPercentTest = QLineEdit(self)
        self.txtPercentTest.setText("20")

        self.lblNeighbourCount = QLabel('Neighbours:')
        self.lblNeighbourCount.adjustSize()

        self.txtNeighbourCount = QLineEdit(self)
        self.txtNeighbourCount.setText("9")

        self.btnExecute = QPushButton("Run Model")
        self.btnExecute.clicked.connect(self.update)

        self.groupBox1Layout.addWidget(self.feature0,0,0,1,1)
        self.groupBox1Layout.addWidget(self.feature1,0,1,1,1)
        self.groupBox1Layout.addWidget(self.feature2,1,0,1,1)
        self.groupBox1Layout.addWidget(self.feature3,1,1,1,1)
        self.groupBox1Layout.addWidget(self.feature4,2,0,1,1)
        self.groupBox1Layout.addWidget(self.feature5,2,1,1,1)
        self.groupBox1Layout.addWidget(self.feature6,3,0,1,1)
        self.groupBox1Layout.addWidget(self.feature7,3,1,1,1)
        self.groupBox1Layout.addWidget(self.feature8, 4, 0,1,1)
        self.groupBox1Layout.addWidget(self.feature9, 4, 1,1,1)
        self.groupBox1Layout.addWidget(self.feature10, 5, 0,1,1)
        self.groupBox1Layout.addWidget(self.feature11, 5, 1,1,1)
        self.groupBox1Layout.addWidget(self.feature12, 6, 0,1,1)
        self.groupBox1Layout.addWidget(self.feature13, 6, 1,1,1)
        self.groupBox1Layout.addWidget(self.feature14, 7, 0,1,1)
        self.groupBox1Layout.addWidget(self.feature15, 7, 1,1,1)
        self.groupBox1Layout.addWidget(self.feature16, 8, 0,1,1)
        self.groupBox1Layout.addWidget(self.feature17, 8, 1,1,1)
        self.groupBox1Layout.addWidget(self.feature18, 9, 0,1,1)
        self.groupBox1Layout.addWidget(self.feature19, 9, 1,1,1)
        self.groupBox1Layout.addWidget(self.feature20, 10, 0,1,1)
        self.groupBox1Layout.addWidget(self.feature21, 10, 1,1,1)
        self.groupBox1Layout.addWidget(self.feature22, 11, 0,1,1)
        self.groupBox1Layout.addWidget(self.feature23, 11, 1,1,1)
        self.groupBox1Layout.addWidget(self.feature24, 12, 0,1,1)
        self.groupBox1Layout.addWidget(self.feature25, 12, 1,1,1)
        self.groupBox1Layout.addWidget(self.feature26, 13, 0,1,1)
        self.groupBox1Layout.addWidget(self.feature27, 13, 1,1,1)
        self.groupBox1Layout.addWidget(self.feature28, 14, 0,1,1)
        self.groupBox1Layout.addWidget(self.feature29, 14, 1,1,1)
        self.groupBox1Layout.addWidget(self.lblPercentTest, 15, 0,1,1)
        self.groupBox1Layout.addWidget(self.txtPercentTest, 15, 1,1,1)
        self.groupBox1Layout.addWidget(self.lblNeighbourCount, 16, 0, 1, 1)
        self.groupBox1Layout.addWidget(self.txtNeighbourCount, 16, 1, 1, 1)
        self.groupBox1Layout.addWidget(self.btnExecute, 17, 0,1,2)

        self.groupBox2 = QGroupBox('Results from the model')
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)
        self.groupBox2.setMinimumSize(400, 50)

        self.lblResults = QLabel('Results:')
        self.lblResults.adjustSize()
        self.txtResults = QPlainTextEdit()
        #self.txtResults.setMinimumSize(200,100)
        self.lblAccuracy = QLabel('Accuracy:')
        self.txtAccuracy = QLineEdit()

        self.groupBox2Layout.addWidget(self.lblResults)
        self.groupBox2Layout.addWidget(self.txtResults)
        self.groupBox2Layout.addWidget(self.lblAccuracy)
        self.groupBox2Layout.addWidget(self.txtAccuracy)


        self.groupBox3 = QGroupBox('Summary and Comparison')
        self.groupBox3Layout = QVBoxLayout()
        self.groupBox3.setLayout(self.groupBox3Layout)
        self.groupBox3.setMinimumSize(400, 50)

        self.lbl_summary = QLabel('Summary:')
        self.lbl_summary.adjustSize()
        self.txt_summary = QPlainTextEdit()
        self.lbl_other_models = QLabel('Other Models Performance:')
        self.other_models = QWidget(self)
        self.other_models.layout = QFormLayout(self.other_models)
        #self.other_modelsLayout = QFormLayout()
        #self.other_models.setLayout(self.other_modelsLayout)
        self.txtAccuracy_rf = QLineEdit()
        self.txtAccuracy_lr = QLineEdit()
        self.txtAccuracy_dt = QLineEdit()
        self.other_models.layout.addRow('Random Forest:', self.txtAccuracy_rf)
        self.other_models.layout.addRow('Logistic Regression:', self.txtAccuracy_lr)
        self.other_models.layout.addRow('Decision Trees:', self.txtAccuracy_dt)

        self.groupBox3Layout.addWidget(self.lbl_summary)
        self.groupBox3Layout.addWidget(self.txt_summary)
        self.groupBox3Layout.addWidget(self.lbl_other_models)
        self.groupBox3Layout.addWidget(self.other_models)


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

        self.groupBoxG2 = QGroupBox('Accuracy vs. K Value')
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

        self.groupBoxG3 = QGroupBox('Cross Validation Score')
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

        self.layout.addWidget(self.groupBox1,0,0,2,1)
        self.layout.addWidget(self.groupBoxG1,0,1,1,1)
        self.layout.addWidget(self.groupBoxG3,0,2,1,1)
        self.layout.addWidget(self.groupBox2,0,3,1,1)
        self.layout.addWidget(self.groupBoxG2,1,1,1,1)
        self.layout.addWidget(self.groupBoxG4,1,2,1,1)
        self.layout.addWidget(self.groupBox3,1,3,1,1)

        self.setCentralWidget(self.main_widget)
        self.resize(1800, 900)
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
        neighbour_input = round(float(self.txtNeighbourCount.text()))

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
        self.clf_knn = KNeighborsClassifier(n_neighbors=neighbour_input)

        # perform training
        self.clf_knn.fit(X_train, y_train)

        #-----------------------------------------------------------------------

        # predicton on test using all features
        y_pred = self.clf_knn.predict(X_test)
        y_pred_score = self.clf_knn.predict_proba(X_test)


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
                self.ax1.text(j, i, str(conf_matrix[i][j]))

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

        ## End Graph1 -- Confusion Matrix

        #::----------------------------------------
        ## Graph 2 - ROC Curve
        #::----------------------------------------

        accuracy_test = []
        accuracy_train = []
        # Might take some time
        for i in range(1, 20,2):
            self.knn_graph = KNeighborsClassifier(n_neighbors=i)
            self.knn_graph.fit(X_train, y_train)
            pred_i = self.knn_graph.predict(X_train)
            accuracy_train.append(accuracy_score(y_train, pred_i))
            pred_i = self.knn_graph.predict(X_test)
            accuracy_test.append(accuracy_score(y_test, pred_i))

        self.ax2.plot(range(1, 20,2),accuracy_train , color='blue', marker='o',
                 markerfacecolor='orange', markersize=5,label="Train Accuracy")
        self.ax2.plot(range(1, 20,2), accuracy_test, color='red', marker='o',
                 markerfacecolor='orange', markersize=5, label="Test Accuracy")
        self.ax2.set_xlabel('K')
        self.ax2.axes.set_xticks(np.arange(1,20,2))
        self.ax2.set_xlabel('K')
        self.ax2.set_ylabel('Accuracy')
        self.ax2.legend(loc="lower right")

        self.fig2.tight_layout()
        self.fig2.canvas.draw_idle()
        ######################################
        # Graph - 3 Feature Importances
        #####################################
        # get feature importances

        scores_arr = []
        #feature_arr = []
        for val in (X_train.columns):
            #print(val)
            cvs_X = X_train[val].values.reshape(-1, 1)
            #print(cvs_X)
            scores = cross_val_score(self.clf_knn, cvs_X, y_train, cv=10)
            scores_arr.append(scores.mean())

        #importances = self.clf_knn.feature_importances_

        # convert the importances into one-dimensional 1darray with corresponding df column names as axis labels
        f_importances = pd.Series(scores_arr, X_train.columns)
        # sort the array in descending order of the importances
        f_importances.sort_values(ascending=False, inplace=True)
        f_importances=f_importances[0:20]
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
        # y_test_bin = label_binarize(y_test, classes=[0, 1])
        # print(pd.get_dummies(y_test))
        # print(pd.get_dummies(y_test).to_numpy())
        y_test_bin = pd.get_dummies(y_test).to_numpy()
        n_classes = y_test_bin.shape[1]

        # From the sckict learn site
        # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        # print(pd.get_dummies(y_test).to_numpy().ravel())

        # print("\n\n********************************\n\n")
        # print(y_pred_score.ravel())
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_pred_score.ravel())

        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        lw = 2
        str_classes= ['No','Yes']
        colors = cycle(['magenta', 'darkorange'])
        for i, color in zip(range(n_classes), colors):
            self.ax4.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='{0} (area = {1:0.2f})'
                           ''.format(str_classes[i], roc_auc[i]))

        self.ax4.plot([0, 1], [0, 1], 'k--', lw=lw)
        self.ax4.set_xlim([0.0, 1.0])
        self.ax4.set_ylim([0.0, 1.05])
        self.ax4.set_xlabel('False Positive Rate')
        self.ax4.set_ylabel('True Positive Rate')
        self.ax4.legend(loc="lower right")

        # show the plot
        self.fig4.tight_layout()
        self.fig4.canvas.draw_idle()

        #::-----------------------------
        # End of graph 4  - ROC curve by class
        #::-----------------------------

        #::-----------------------------------------------------
        # Other Models Comparison
        #::-----------------------------------------------------

        self.other_clf_rf=RandomForestClassifier(n_estimators=100, random_state=500)
        self.other_clf_rf.fit(X_train, y_train)
        y_pred_rf = self.other_clf_rf.predict(X_test)
        self.accuracy_rf=accuracy_score(y_test, y_pred_rf) *100
        self.txtAccuracy_rf.setText(str(self.accuracy_rf))

        self.other_clf_dt = DecisionTreeClassifier(criterion="gini")
        self.other_clf_dt.fit(X_train, y_train)
        y_pred_dt = self.other_clf_dt.predict(X_test)
        self.accuracy_dt = accuracy_score(y_test, y_pred_dt) * 100
        self.txtAccuracy_dt.setText(str(self.accuracy_dt))

        self.other_clf_lr = LogisticRegression(random_state=500)
        self.other_clf_lr.fit(X_train, y_train)
        y_pred_lr = self.other_clf_lr.predict(X_test)
        self.accuracy_lr = accuracy_score(y_test, y_pred_lr) * 100
        self.txtAccuracy_lr.setText(str(self.accuracy_lr))

    #::-----------------------------
        # End of Other Models Comparison
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

        EDA1Button = QAction(QIcon('analysis.png'),'Variable Distribution', self)
        EDA1Button.setStatusTip('Presents the variable distribution')
        EDA1Button.triggered.connect(self.EDA1)
        EDAMenu.addAction(EDA1Button)

        EDA2Button = QAction(QIcon('analysis.png'), 'Variable Relation', self)
        EDA2Button.setStatusTip('Presents the relationship between variables')
        EDA2Button.triggered.connect(self.EDA2)
        EDAMenu.addAction(EDA2Button)

        EDA4Button = QAction(QIcon('analysis.png'), 'Attrition Relation', self)
        EDA4Button.setStatusTip('Compares the variables with respect to Attrition')
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
        MLModel1Button =  QAction(QIcon(), 'Decision Tree', self)
        MLModel1Button.setStatusTip('ML algorithm ')
        MLModel1Button.triggered.connect(self.MLDT)

        #::------------------------------------------------------
        # Random Forest Classifier
        #::------------------------------------------------------
        MLModel2Button = QAction(QIcon(), 'Random Forest Classifier', self)
        MLModel2Button.setStatusTip('Random Forest Classifier ')
        MLModel2Button.triggered.connect(self.MLRF)

        #::--------------------------------------------------
        # Logistic REgression Model
        #::--------------------------------------------------
        MLModel3Button = QAction(QIcon(), 'Logistic Regression', self)
        MLModel3Button.setStatusTip('Logistic Regression')
        MLModel3Button.triggered.connect(self.MLLR)

        #::--------------------------------------------------
        # KNN Model
        #::--------------------------------------------------
        MLModel4Button = QAction(QIcon(), 'K- Nearest Neigbor', self)
        MLModel4Button.setStatusTip('K- Nearest Neigbor')
        MLModel4Button.triggered.connect(self.MLKNN)

        MLModelMenu.addAction(MLModel1Button)
        MLModelMenu.addAction(MLModel2Button)
        MLModelMenu.addAction(MLModel3Button)
        MLModelMenu.addAction(MLModel4Button)

        self.dialogs = list()

    def EDA1(self):
        #::------------------------------------------------------
        # Creates the histogram
        # The X variable contains the happiness.score
        # X was populated in the method data_happiness()
        # at the start of the application
        #::------------------------------------------------------
        dialog = VariableDistribution()
        self.dialogs.append(dialog)
        dialog.show()

    def EDA2(self):
        #::---------------------------------------------------------
        # This function creates an instance of HappinessGraphs class
        # This class creates a graph using the features in the dataset
        # happiness vrs the score of happiness
        #::---------------------------------------------------------
        dialog = VariableRelation()
        self.dialogs.append(dialog)
        dialog.show()

    def EDA4(self):
        #::----------------------------------------------------------
        # This function creates an instance of the CorrelationPlot class
        #::----------------------------------------------------------
        dialog = AttritionRelation()
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

    def MLLR(self):
        #::-----------------------------------------------------------
        # This function creates an instance of the DecisionTree class
        # This class presents a dashboard for a Decision Tree Algorithm
        # using the happiness dataset
        #::-----------------------------------------------------------
        dialog = LogisticRegressionClassifier()
        self.dialogs.append(dialog)
        dialog.show()

    def MLKNN(self):
        #::-----------------------------------------------------------
        # This function creates an instance of the DecisionTree class
        # This class presents a dashboard for a Decision Tree Algorithm
        # using the happiness dataset
        #::-----------------------------------------------------------
        dialog = KNNClassifier()
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
    global personal_features
    global organisation_features
    global commution_features
    global satisfaction_features
    global continuous_features
    global categorical_features

    #happiness = pd.read_csv('2017.csv')
    #X= happiness["Happiness.Score"]
    #y= happiness["Country"]
    attr_data = pd.read_csv('HR-Employee-Attrition.csv')
    all_columns = attr_data.columns.tolist()
    #print(all_columns)
    all_columns.remove("Attrition")
    features_list=all_columns.copy()
    #print(features_list)
    target_variable="Attrition"
    class_names = ['No', 'Yes']
    label_encoder_variables =["Education","JobLevel"]
    hot_encoder_variables=["BusinessTravel","Department","EducationField","Gender","JobRole","MaritalStatus","OverTime","StockOptionLevel"]
    personal_features=["Age","Education","EducationField","MaritalStatus","Gender"]
    organisation_features=["DailyRate","Department","HourlyRate","JobInvolvement","JobLevel","JobRole","MonthlyIncome","MonthlyRate","NumCompaniesWorked","OverTime","PercentSalaryHike","PerformanceRating","StockOptionLevel","TotalWorkingYears","TrainingTimesLastYear","YearsAtCompany","YearsInCurrentRole","YearsSinceLastPromotion","YearsWithCurrManager"]
    commution_features=["BusinessTravel","DistanceFromHome"]
    satisfaction_features=["EnvironmentSatisfaction","JobSatisfaction","RelationshipSatisfaction","WorkLifeBalance"]
    continuous_features=["Age","DistanceFromHome","DailyRate","HourlyRate","MonthlyIncome","MonthlyRate","NumCompaniesWorked","PercentSalaryHike","TotalWorkingYears","TrainingTimesLastYear","YearsAtCompany","YearsInCurrentRole","YearsSinceLastPromotion","YearsWithCurrManager"]
    categorical_features=list(set(features_list) - set(continuous_features))

if __name__ == '__main__':
    #::------------------------------------
    # First reads the data then calls for the application
    #::------------------------------------
    attrition_data()
    main()
