#when we import hydralit, we automatically get all of Streamlit
import hydralit as hy
from ctypes import alignment
import streamlit as st
from PIL import Image
import pandas
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from functions import findk,findPrediction
from ctypes import alignment
import streamlit as st
from PIL import Image
import pandas
import matplotlib.pyplot as plt
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from PIL import ImageFilter
from functions import findk,findPrediction
import cv2

app = hy.HydraApp(title='Simple Multi-Page App')


@app.addapp()
def My_Home():
 exec(open("home.py").read())

@app.addapp()
def Iris_Interface():
 exec(open("interface.py").read())

@app.addapp()
def Digit_Interface():
 exec(open("interface2.py").read())

@app.addapp()
def K_Interface():
 exec(open("k.py").read())

#Run the whole lot, we get navbar, state management and app isolation, all with this tiny amount of work.
app.run()