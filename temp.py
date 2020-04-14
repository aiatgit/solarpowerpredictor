from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model
from tkinter import *
from tkinter.filedialog import askopenfilename
from pandas import read_csv
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from tkinter import ttk
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplot, figure, subplots
from math import sqrt
import matplotlib
matplotlib.use("TkAgg")

LARGE_FONT = ("Verdana", 12)
x1 = 0
x2 = 0
y1 = 0
y2 = 0
x = 0
y = 0
dict1 = {}


def import_csv_data():

    csv_file_path = askopenfilename()
    global x1, x2, y1, y2, x, y
    data = pd.read_csv(csv_file_path)
    x = data.iloc[:, 1]
    y = data.iloc[:, 2]
    train_size = int(len(x) * 0.80)
    x1, x2 = x[0:train_size], x[train_size:]
    y1, y2 = y[0:train_size], y[train_size:]

    x1 = pd.DataFrame(x1)
    x2 = pd.DataFrame(x2)
    y1 = pd.DataFrame(y1)
    y2 = pd.DataFrame(y2)

    sc = MinMaxScaler(feature_range=(0, 1))
    x1 = sc.fit_transform(x1)
    x2 = sc.transform(x2)

    y1 = sc.fit_transform(y1)
    y2 = sc.transform(y2)


def plotgraph(y_pred,name,error,score):
    error = round(error, 6)
    error = str(error)
    score = round(score, 6)
    score = str(score)
    plt.plot(y_pred)
    plt.title("Result of " + name + "\n" + "error = " + error + "   MSE = " + score)
    plt.xlabel("Time")
    plt.ylabel("Solar Power generation")

    plt.show()


def compare2():

    f1, axes = subplots(2, 2)

    model = RandomForestRegressor()
    name = "Random Forest"
    model.fit(x1, y1)
    y_pred = model.predict(x2)
    error = mean_squared_error(y2, y_pred)
    score = model.score(y2, y_pred)
    error = round(error, 6)
    error = str(error)
    score = round(score, 6)
    score = str(score)
    axes[0, 0].plot(y_pred)
    axes[0, 0].set_title(name + ": error = " + error + "   MSE = " + score)

    model = GradientBoostingRegressor()
    name = "Gradient Booster"
    model.fit(x1, y1)
    y_pred = model.predict(x2)
    error = mean_squared_error(y2, y_pred)
    score = model.score(y2, y_pred)
    error = round(error, 6)
    error = str(error)
    score = round(score, 6)
    score = str(score)
    axes[0, 1].plot(y_pred)
    axes[0, 1].set_title( name + ": error = " + error + "   MSE = " + score)

    model = ExtraTreesRegressor()
    name = "Extra Trees Regressor"
    model.fit(x1, y1)
    y_pred = model.predict(x2)
    error = mean_squared_error(y2, y_pred)
    score = model.score(y2, y_pred)
    error = round(error, 6)
    error = str(error)
    score = round(score, 6)
    score = str(score)
    axes[1, 0].plot(y_pred)
    axes[1, 0].set_title(name + ": error = " + error + "   MSE = " + score)

    model = AdaBoostRegressor()
    name = "AdaBoost Regressor"
    model.fit(x1, y1)
    y_pred = model.predict(x2)
    error = mean_squared_error(y2, y_pred)
    score = model.score(y2, y_pred)
    score = model.score(y2, y_pred)
    error = round(error, 6)
    error = str(error)
    score = round(score, 6)
    score = str(score)
    axes[1, 1].plot(y_pred)
    axes[1, 1].set_title(name + ": error = " + error + "   MSE = " + score)

    f1.show()


def compare1():

    f1,axes = subplots(2, 3)

    model = LinearRegression()
    name = "Linear Regression"
    model.fit(x1, y1)
    y_pred = model.predict(x2)
    error = mean_squared_error(y2, y_pred)
    score = model.score(y_pred, y2)
    error = round(error, 6)
    error = str(error)
    score = round(score, 6)
    score = str(score)
    axes[0,0].plot(y_pred)
    axes[0,0].set_title(name + ": error = " + error + "   MSE = " + score)


    model = joblib.load('lasso.pkl')
    y_pred = model.predict(x2)
    name = "Lasso"
    error = mean_squared_error(y2, y_pred)
    score = model.score(y2, y_pred)
    error = round(error, 6)
    error = str(error)
    score = round(score, 6)
    score = str(score)
    axes[0, 1].plot(y_pred)
    axes[0, 1].set_title(name +": error = " + error + "   MSE = " + score)

    model = KNeighborsRegressor()
    name = "KNN Regression"
    model.fit(x1, y1)
    y_pred = model.predict(x2)
    error = mean_squared_error(y2, y_pred)
    score = model.score(y2, y_pred)
    error = round(error, 6)
    error = str(error)
    score = round(score, 6)
    score = str(score)
    axes[0, 2].plot(y_pred)
    axes[0, 2].set_title(name +": error = " + error + "   MSE = " + score)

    model = DecisionTreeRegressor()
    name = "Decision Tree"
    model.fit(x1, y1)
    y_pred = model.predict(x2)
    error = mean_squared_error(y2, y_pred)
    score = model.score(y2, y_pred)
    error = round(error, 6)
    error = str(error)
    score = round(score, 6)
    score = str(score)
    axes[1, 0].plot(y_pred)
    axes[1, 0].set_title(name + ": error = " + error + "   MSE = " + score)

    model = SVR()
    name = "SVR"
    model.fit(x1, y1)
    y_pred = model.predict(x2)
    error = mean_squared_error(y2, y_pred)
    score = model.score(y2, y_pred)
    error = round(error, 6)
    error = str(error)
    score = round(score, 6)
    score = str(score)
    axes[1, 1].plot(y_pred)
    axes[1, 1].set_title(name + ": error = " + error + "   MSE = " + score)

    model = ElasticNet()
    name = "Elastic Net"
    model.fit(x1, y1)
    y_pred = model.predict(x2)
    error = mean_squared_error(y2, y_pred)
    score = model.score(y2, y_pred)
    error = round(error, 6)
    error = str(error)
    score = round(score, 6)
    score = str(score)
    axes[1, 2].plot(y_pred)
    axes[1, 2].set_title(name + ": error = " + error + "   MSE = " + score)

    f1.show()


def BOP():

    big_score=0
    big_name=""
    big_y_pred=0
    big_error=0

    model = LinearRegression()
    name = "Linear Regression"
    model.fit(x1, y1)
    y_pred = model.predict(x2)
    error = mean_squared_error(y2, y_pred)
    score = model.score(y_pred, y2)

    if(score>big_score and score is not 1):
        big_score = score
        big_name = name
        big_y_pred = y_pred
        big_error = error


    model = KNeighborsRegressor()
    name = "KNN Regression"
    model.fit(x1, y1)
    y_pred = model.predict(x2)
    error = mean_squared_error(y2, y_pred)
    score = model.score(y2, y_pred)

    if (score > big_score and score is not 1):
        big_score = score
        big_name = name
        big_y_pred = y_pred
        big_error = error

    model = DecisionTreeRegressor()
    name = "Decision Tree"
    model.fit(x1, y1)
    y_pred = model.predict(x2)
    error = mean_squared_error(y2, y_pred)
    score = model.score(y2, y_pred)

    if (score > big_score and score is not 1):
        big_score = score
        big_name = name
        big_y_pred = y_pred
        big_error = error

    model = SVR()
    name = "SVR"
    model.fit(x1, y1)
    y_pred = model.predict(x2)
    error = mean_squared_error(y2, y_pred)
    score = model.score(y2, y_pred)

    if (score > big_score and score is not 1):
        big_score = score
        big_name = name
        big_y_pred = y_pred
        big_error = error


    model = RandomForestRegressor()
    name = "Random Forest"
    model.fit(x1, y1)
    y_pred = model.predict(x2)
    error = mean_squared_error(y2, y_pred)
    score = model.score(y2, y_pred)

    if (score > big_score and score is not 1):
        big_score = score
        big_name = name
        big_y_pred = y_pred
        big_error = error

    model = GradientBoostingRegressor()
    name = "Gradient Booster"
    model.fit(x1, y1)
    y_pred = model.predict(x2)
    error = mean_squared_error(y2, y_pred)
    score = model.score(y2, y_pred)

    if (score > big_score and score is not 1):
        big_score = score
        big_name = name
        big_y_pred = y_pred
        big_error = error

    model = ExtraTreesRegressor()
    name = "Extra Trees Regressor"
    model.fit(x1, y1)
    y_pred = model.predict(x2)
    error = mean_squared_error(y2, y_pred)
    score = model.score(y2, y_pred)

    if (score > big_score and score is not 1):
        big_score = score
        big_name = name
        big_y_pred = y_pred3
        big_error = error

    model = AdaBoostRegressor()
    name = "AdaBoost Regressor"
    model.fit(x1, y1)
    y_pred = model.predict(x2)
    error = mean_squared_error(y2, y_pred)
    score = model.score(y2, y_pred)

    if (score > big_score and score is not 1):
        big_score = score
        big_name = name
        big_y_pred = y_pred
        big_error = error

    plotgraph(big_y_pred, big_name, error, big_score)



def LR():
    global x1,x2,y1,y2,dict1
    model = LinearRegression()
    name="Linear Regression"
    model.fit(x1, y1)
    y_pred = model.predict(x2)
    error = mean_squared_error(y2, y_pred)
    score = model.score(y_pred, y2)
    dict1[name]=score
    plotgraph(y_pred,name, error, score)


def Lasso1():
    global x1,x2,y1,y2,dict1

    model = joblib.load('lasso.pkl')
    y_pred = model.predict(x2)
    name="Lasso"
    error = mean_squared_error(y2, y_pred)
    score = model.score(y2, y_pred)
    plotgraph(y_pred, name, error, score)

def KNN():
    global x1,x2,y1,y2,dict1
    model = KNeighborsRegressor()
    name="KNN Regression"
    model.fit(x1, y1)
    y_pred = model.predict(x2)
    error = mean_squared_error(y2, y_pred)
    score = model.score(y2,y_pred)
    dict1[name] = score
    plotgraph(y_pred, name, error, score)

def DT():
    global x1,x2,y1,y2,dict1
    model = DecisionTreeRegressor()
    name="Decision Tree"
    model.fit(x1, y1)
    y_pred = model.predict(x2)
    error = mean_squared_error(y2, y_pred)
    score = model.score(y2,y_pred)
    dict1[name] = score
    plotgraph(y_pred, name, error, score)

def SVR1():
    global x1,x2,y1,y2,dict1
    model = SVR()
    name="SVR"
    model.fit(x1, y1)
    y_pred = model.predict(x2)
    error = mean_squared_error(y2, y_pred)
    score = model.score(y2,y_pred)
    dict1[name] = score
    plotgraph(y_pred, name, error, score)

def Elastic():
    global x1,x2,y1,y2,dict1
    model = ElasticNet()
    name="Elastic Net"
    model.fit(x1, y1)
    y_pred = model.predict(x2)
    error = mean_squared_error(y2, y_pred)
    score = model.score(y2,y_pred)
    plotgraph(y_pred, name, error, score)


# ensemble learning


def RF():
    global x1, x2, y1, y2,dict1
    model = RandomForestRegressor()
    name = "Random Forest"
    model.fit(x1, y1)
    y_pred = model.predict(x2)
    error = mean_squared_error(y2, y_pred)
    score = model.score(y2,y_pred)
    dict1[name] = score
    plotgraph(y_pred, name, error, score)


def GB():
    global x1, x2, y1, y2,dict1
    model = GradientBoostingRegressor()
    name = "Gradient Booster"
    model.fit(x1, y1)
    y_pred = model.predict(x2)
    error = mean_squared_error(y2, y_pred)
    score = model.score(y2,y_pred)
    dict1[name] = score
    plotgraph(y_pred, name, error, score)


def ET():
    global x1, x2, y1, y2,dict1
    model = ExtraTreesRegressor()
    name = "Extra Trees Regressor"
    model.fit(x1, y1)
    y_pred = model.predict(x2)
    error = mean_squared_error(y2, y_pred)
    score = model.score(y2,y_pred)
    dict1[name] = score
    plotgraph(y_pred, name, error, score)


def ADB():
    global x1, x2, y1, y2,dict1
    model = AdaBoostRegressor()
    name = "AdaBoost Regressor"
    model.fit(x1, y1)
    y_pred = model.predict(x2)
    error = mean_squared_error(y2, y_pred)
    score = model.score(y2,y_pred)
    dict1[name] = score
    plotgraph(y_pred, name, error, score)

def Arima1():
    global x1,x2,y1,y2
    Actual = [i for i in x1]
    Predictions = list()

    for timepoint in range(len(x2)):
        ActualValue = x2[timepoint]
        # forcast value
        model = ARIMA(Actual, order=(3, 1, 0))
        model_fit = model.fit(disp=0)
        Prediction = model_fit.forecast()[0]
        # add it in the list
        Predictions.append(Prediction)
        Actual.append(ActualValue)

    plt.plot(Predictions, color='red')
    plt.show()


def Lstm():
    X_test = x2[:-1]
    y_test = x2[1:]

    X_train = x1[:-1]
    y_train = x2[1:]

    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    model = load_model('my_model.h5')

    x_pred = model.predict(X_test)

    error = mean_squared_error(x2[:-1], x_pred)

    error = str(error)
    plt.title("Result of LSTM" + "\nMSE = "+ error)
    plt.xlabel("Time")
    plt.ylabel("Solar Power generation")
    plt.plot(x_pred, color="red")
    plt.show()


def ann():
    global dict1
    model = load_model("ann.h5")

    y_pred = model.predict(x2)
    error = mean_squared_error(y2, y_pred)
    plt.plot(y_pred)
    plt.title("Artificial neural network")
    plt.xlabel("Time")
    plt.ylabel("Solar Power generation")
    plt.show()



class printdata(Frame):

    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        label = Label(self, text="Print", font=LARGE_FONT)
        label.pack(padx=7,pady=7)




        button1 = ttk.Button(self, text="Solar Irradiance Training",
                         command=x1_print)
        button1.pack(padx=7,pady=7)

        button1 = ttk.Button(self, text="Solar Irradiance Testing",
                             command=x2_print)
        button1.pack(padx=7,pady=7)

        button1 = ttk.Button(self, text="Solar Power Training",
                             command=y1_print)
        button1.pack(padx=7,pady=7)

        button1 = ttk.Button(self, text="Solar Power Testing",
                             command=y2_print)
        button1.pack(padx=7,pady=7)

        button0 = ttk.Button(self, text="back to home page",
                             command=lambda: controller.show_frame(startpage))
        button0.pack(padx=7, pady=7)



class SPP(Tk):


    def __init__(self, *args, **kwargs):
        Tk.__init__(self, *args, **kwargs)
        container = Frame(self, bg="blue", width=500, height=300)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (startpage, printdata, machine_learning, ensemble_learning, deep_learning):

            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(startpage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


class startpage(Frame):

    def __init__(self, parent, controller):
        Frame.__init__(self, parent)

        toolbar = Frame(self, bg="blue", width=500, height=300)
        label = Label(toolbar, text="Start page", font=LARGE_FONT)
        label.pack(padx=8, pady=8)
        toolbar.pack()


        insertbutt1 = ttk.Button(self, text="Load Dataset", command=import_csv_data)
        insertbutt1.pack(side=TOP, padx=5, pady=5)

        printbutt = ttk.Button(self, text="Print",
                               command = lambda: controller.show_frame(printdata))
        printbutt.pack(side=TOP, padx=5, pady=5)




        toolbar1 = Frame(self, bg="blue", width=500, height=300)

        label1 = ttk.Label(toolbar1, text="Forecasting Algorithms", font=LARGE_FONT)

        label1.pack(padx=8, pady=8)

        toolbar1.pack(side=TOP)

        button1 = ttk.Button(self,text="LSTM",

                         command = Lstm)
        button1.pack(padx=5, pady=5)

        button2 = ttk.Button(self, text="ARIMA",
                         command = Arima1)
        button2.pack(padx=5, pady=5)



        toolbar2 = Frame(self, bg="blue", width=500, height=300)

        label2 = ttk.Label(toolbar2, text="Prediction Algorithms", font=LARGE_FONT)

        label2.pack(padx=8, pady=8)

        toolbar2.pack(side=TOP)

        label3 = ttk.Label(self, text="Using Machine learning:", font=LARGE_FONT)

        label3.pack(side=TOP, padx=5, pady=5)

        button3 = ttk.Button(self, text="machine learning",
                             command=lambda: controller.show_frame(machine_learning))
        button3.pack(side=TOP, padx=5, pady=5)

        label4 = ttk.Label(self, text="Using Ensemble learning:", font=LARGE_FONT)

        label4.pack(side=TOP, padx=2 ,pady= 2)

        button5 = ttk.Button(self, text="Ensemble learning",
                             command=lambda: controller.show_frame(ensemble_learning))
        button5.pack(side=TOP, padx=2 ,pady= 2)

        label5 = ttk.Label(self, text="Using Deep learning:", font=LARGE_FONT)

        label5.pack(side=TOP, padx=2, pady=2)

        button6 = ttk.Button(self, text="Deep learning",
                             command=lambda: controller.show_frame(deep_learning))
        button6.pack(side=TOP, padx=2, pady=2)

        button7 = ttk.Button(self, text="Best of Prediction",
                             command=BOP)
        button7.pack(side=TOP, padx=2, pady=2)


class machine_learning(Frame):

    def __init__(self, parent, controller):
        Frame.__init__(self, parent)

        toolbar2 = Frame(self, bg="blue", width=500, height=300)
        label = Label(toolbar2, text="Machine learning algorithms", font=LARGE_FONT)
        label.pack(padx=7,pady=7)
        toolbar2.pack()

        button1 = ttk.Button(self, text="Linear Regression ",
                            command=LR)
        button1.pack(padx=7,pady=7)

        button2 = ttk.Button(self, text="Lasso Regression ",
                            command=Lasso1)
        button2.pack(padx=7,pady=7)

        button3 = ttk.Button(self, text="K nearest neighbours ",
                            command=KNN)
        button3.pack(padx=7,pady=7)

        button4 = ttk.Button(self, text="Decision tree",
                            command=DT)
        button4.pack(padx=7,pady=7)

        button5 = ttk.Button(self, text="SVR",
                            command=SVR1)
        button5.pack(padx=7,pady=7)

        button6 = ttk.Button(self, text="Elastic net",
                            command=Elastic)
        button6.pack(padx=7,pady=7)

        button7 = ttk.Button(self, text="Compare",
                             command=compare1)
        button7.pack(padx=7, pady=7)

        button = ttk.Button(self, text="back to home page",
                         command=lambda: controller.show_frame(startpage))
        button.pack(padx=7,pady=7)


class ensemble_learning(Frame):

    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        label = Label(self, text="Ensemble learning algorithms", font=LARGE_FONT)
        label.pack(padx=7,pady=7)

        button1 = ttk.Button(self, text="Random Forest",
                            command=RF)
        button1.pack(padx=7,pady=7)

        button2 = ttk.Button(self, text="Gradient Booster",
                            command=GB)
        button2.pack(padx=7,pady=7)

        button3 = ttk.Button(self, text="Extra Tree",
                            command=ET)
        button3.pack(padx=7,pady=7)

        button4 = ttk.Button(self, text="Adabo booster",
                            command=ADB)
        button4.pack(padx=7,pady=7)

        button5 = ttk.Button(self, text="Compare",
                             command=compare2)
        button5.pack(padx=7,pady=7)

        button = ttk.Button(self, text="back to home page",
                       command=lambda: controller.show_frame(startpage))
        button.pack(padx=7,pady=7)


class deep_learning(Frame):

    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        label = Label(self, text="Deep learning algorithms", font=LARGE_FONT)
        label.pack(padx=7,pady=7)

        button1 = ttk.Button(self, text="Artificial Neural network",
                            command=ann)
        button1.pack(padx=7,pady=7)

        button = ttk.Button(self, text="back to home page",
                            command=lambda: controller.show_frame(startpage))
        button.pack(padx=7,pady=7)

def x1_print():

    global plt,x1
    plt.plot(x1)
    plt.title("Solar Irradiance Training")
    plt.xlabel("Time")
    plt.ylabel("Solar Irradiance")
    plt.show()


def x2_print():

    global plt,x2
    plt.plot(x2)
    plt.title("Solar Irradiance Testing")
    plt.xlabel("Time")
    plt.ylabel("Solar Irradiance")
    plt.show()


def y1_print():

    global plt,y1
    plt.plot(y1)
    plt.title("Solar Power Training")
    plt.xlabel("Time")
    plt.ylabel("Solar Power")
    plt.show()


def y2_print():

    global plt,y2
    plt.plot(y2)
    plt.title("Solar Power Testing")
    plt.xlabel("Time")
    plt.ylabel("Solar Power")
    plt.show()


app = SPP()
app.mainloop()
