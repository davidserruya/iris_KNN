#when we import hydralit, we automatically get all of Streamlit
import hydralit as hy
from ctypes import alignment
import streamlit as st
from PIL import Image
import pandas
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import subprocess
import os
from functions import *
import streamlit as st
from streamlit_option_menu import option_menu

with st.expander("MENU"):
     menu = option_menu(None, ["Home Interface", "Iris Interface", "Digit Interface", 'K Interface'], 
    icons=['house', "suit-club", "123", 'graph-up'], 
    menu_icon="cast", default_index=0, orientation="vertical")


if menu == 'Home Interface':
    exec(open("home.py").read())    

elif menu == 'Iris Interface':
    exec(open("interface.py").read())
    

elif menu == 'Digit Interface':
    exec(open("interface2.py").read())

elif menu == 'K Interface':
    exec(open("k.py").read())
