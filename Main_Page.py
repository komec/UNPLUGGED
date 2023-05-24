import streamlit as st
import pandas as pd
import plotly.express as px
import base64
from pathlib import Path
from PIL import Image
import plotly.figure_factory as ff
import numpy as np
#from streamlit_extras.add_vertical_space import add_vertical_space
#from streamlit_extras.colored_header import colored_header

primaryColor="#6eb52f"
backgroundColor="#f0f0f5"
secondaryBackgroundColor="#e0e0ef"
textColor="#262730"
font="sans serif"


def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

# Page configs 
# https://www.webfx.com/tools/emoji-cheat-sheet/                      https://emojipedia.org/symbols/
st.set_page_config(page_title="Unplugged",
                    page_icon="files/logo.png",
                    layout="wide"
                    )

# Remove Streamlit footer and main page  --- custom css code
hide_st_style = """ <style> #MainMenu {visibility: hidden;}
                footer {visibility: hidden;} 
                
                </style> 
                """

# header {visibility: hidden;}   Add to delete header

st.markdown(hide_st_style, unsafe_allow_html=True)




#########################
# Sidebar containers
#########################
sidebar_entry_container = st.sidebar.container()
sidebar_import_container = st.sidebar.container()
sidebar_about_container = st.sidebar.container()



#########################
# Import sidebar logo
#########################
sidebar_entry_container.markdown('''<center><img src='data:image/png;base64,{}' class='img-fluid' width=192 height=192></center>'''.format(img_to_bytes("files/logomain.jpeg")), unsafe_allow_html=True)

# Sidebar text
# Title
sidebar_entry_container.title(" \t UNPLUGGED by komec")

# Info box
sidebar_entry_container.info("Machine learning application for seismicity")


# Add vertical space
#with sidebar_entry_container:
#    add_vertical_space(1)



#########################
# Add info box
#########################
# Add vertical space
#with sidebar_about_container:
#    add_vertical_space(8)


sidebar_about_container.title("About")
sidebar_about_container.info(
                                """
                                GitHub repo:  <https://github.com/>
                                """
                                )


#########################
# Import section
#########################
file = sidebar_import_container.file_uploader("Choose a CSV file",type="csv", accept_multiple_files=False)



#########################
# Main Page 
#########################
#colored_header(
#                label="P-wave Prediction Modelling",
#                description="P-wave prediction modelling for Simav-Kütahya earthquake activity.",
#                color_name="light-blue-70",
#                )



#########################
seisinfo_container = st.container()
fileinfo_container = st.container()
results_container = st.container()

#########################
seisinfo_container.header("P-wave Prediction Modelling")
seisinfo_container.caption("___P-wave prediction modelling for Simav-Kütahya earthquake activity.___")

seisinfo_container.subheader("Seismicity Analysis for Simav Region")

figcol1, figcol2, figcol3  = seisinfo_container.columns(3)
figcol4, figcol5, figcol6 =  seisinfo_container.columns(3)


image1 = Image.open('files/stn-map.jpg')
figcol1.image(image1, caption='Sunrise by the mountains')

image2 = Image.open('files/Dept-Vel-Cross_HYPODD1.jpg')
figcol2.image(image2, caption='Sunrise by the mountains')

image3 = Image.open('files/topo_fay.jpg')
figcol3.image(image3, caption='Sunrise by the mountains')


image4 = Image.open('files/fig1.jpeg')
figcol4.image(image4, caption='Sunrise by the mountains')

image5 = Image.open('files/fig2.jpeg')
figcol5.image(image5, caption='Sunrise by the mountains')

image6 = Image.open('files/fig3.jpeg')
figcol6.image(image6, caption='Sunrise by the mountains')



#########################
fileinfo_container.subheader("Pre-data Processing")
fileinfo_container.caption("___Information about the uploaded dataset.___")


results_container.header("Feature Engineering")

###
st.session_state["pbar"] = 0
prog_bar = results_container.progress(st.session_state["pbar"])
###


if not file:
    if "uploaded_files" not in st.session_state:
        pass
    else:
        del st.session_state["uploaded_files"]
        del st.session_state["pbar"]
else:
    st.session_state["uploaded_files"] = file


# Control file upload
if "uploaded_files" not in st.session_state:
    fileinfo_container.warning("___Upload file to start___")

else:
    # Continue if file is 
    fileinfo_container.success("___File uploaded succesfully___")

    # Read dataset
    df_data = pd.read_csv(file)


    #########################
    # Calculate Travel Time and Vp
    #########################
    def travel_time(Hour, Min, Sec, StnHour, StnMin, StnSec):
        return (StnHour - Hour) * 3600 + (StnMin - Min) * 60 + (StnSec - Sec)

    df_data['TravelTime'] = travel_time(df_data['Hour'], df_data['Min'], df_data['Sec'], df_data['StnHour'], df_data['StnMin'], df_data['StnSec'])

    df_data['Vp'] = df_data['Distance'] / df_data['TravelTime']

    ###
    st.session_state["pbar"] = st.session_state["pbar"] + 10
    prog_bar.progress(st.session_state["pbar"])
    ###

    
    # Create expander
    info_expander = fileinfo_container.expander("Show Dataset", expanded=False)

    details_expander = fileinfo_container.expander("EDA (Exploratory data analysis)", expanded=False)

    #########################
    # Dataset Info
    #########################
    options = df_data.columns.tolist()
    selected_options = info_expander.multiselect("Select Columns", options, default=options)
    filtered_df = df_data[selected_options]  

    info_expander.dataframe(filtered_df)
    
    ###
    st.session_state["pbar"] = st.session_state["pbar"] + 10
    prog_bar.progress(st.session_state["pbar"])
    ###


    #########################
    # Dataset details
    #########################
    details_expander.caption("___1. Shape___")
    details_expander.caption(f" \
                           Number of Rows:  {df_data.shape[0]} \n\n  \
                           Number of Columns:  {df_data.shape[1]}\n\n \
                           " )

    details_expander.caption("___2. NA___")
    details_expander.dataframe(df_data.isnull().sum().T)

    details_expander.caption("___3. Quantiles___")
    details_expander.dataframe(df_data.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
    
    ###
    st.session_state["pbar"] = st.session_state["pbar"] + 10
    prog_bar.progress(st.session_state["pbar"])
    ###

    
    #########################
    # Figures - Analysis Results
    #########################
    # Figure containers
    fig_corr_container = results_container.container()
    fig_outl_container = results_container.container()
    fig_dens_container = results_container.container()
    fig_model_container = results_container.container()
    fig_featimp_container = results_container.container()

    
    df_data = df_data.dropna()

    #########################
    # Correlation Figure
    #########################
    df_corr = df_data.corr()
    fig_corr_container.subheader("Correlation Matrix")
    fig_corr = px.imshow(df_corr, aspect="auto")

    fig_corr.update_layout(width= 1440, height=960)    
    fig_corr.update_traces(textfont_size=8.25)

    fig_corr_container.plotly_chart(fig_corr, theme="streamlit", use_container_width=True)
    
    ###
    st.session_state["pbar"] = st.session_state["pbar"] +10
    prog_bar.progress(st.session_state["pbar"])
    ###

    #########################
    # Figures - Outliers
    #########################
    fig_corr_container.subheader("Outliers")

    figcol1, figcol2, figcol3  = fig_outl_container.columns(3)
    figcol4, figcol5, figcol6  = fig_outl_container.columns(3)


    image1 = Image.open('files/Depth_boxplot.png')
    figcol1.image(image1)

    image2 = Image.open('files/Residual_boxplot.png')
    figcol2.image(image2)

    image3 = Image.open('files/GAP_boxplot.png')
    figcol3.image(image3)

    image4 = Image.open('files/Distance_boxplot.png')
    figcol4.image(image4)

    image5 = Image.open('files/LonError_boxplot.png')
    figcol5.image(image5)
    
    image6 = Image.open('files/LatError_boxplot.png')
    figcol6.image(image6)

    ###
    st.session_state["pbar"] = st.session_state["pbar"] +10
    prog_bar.progress(st.session_state["pbar"])
    ###

    #########################
    # Figures - Density
    #########################
    fig_dens_container.subheader("Data Visualization")

    figcol1, figcol2, figcol3  = fig_dens_container.columns(3)
    figcol4, figcol5, figcol6  = fig_dens_container.columns(3)

    image1 = Image.open('files/Depth_density.png')
    figcol1.image(image1)

    image2 = Image.open('files/Residual_density.png')
    figcol2.image(image2)

    image3 = Image.open('files/Magnitude_density.png')
    figcol3.image(image3)

    image4 = Image.open('files/GAP_density.png')
    figcol4.image(image4)

    image5 = Image.open('files/Distance_density.png')
    figcol5.image(image5)
    
    image6 = Image.open('files/Azimuth_density.png')
    figcol6.image(image6)

    ###
    st.session_state["pbar"] = st.session_state["pbar"] +10
    prog_bar.progress(st.session_state["pbar"])
    ###

    #########################
    # Figures - Model
    #########################
    fig_model_container.subheader("Model Output Metrices")
    
    figcol1, figcol2, figcol3  = fig_model_container.columns(3)

    image1 = Image.open('files/acc.png')
    figcol1.image(image1)
    
    image2 = Image.open('files/roc.png')
    figcol2.image(image2)
    
    image3 = Image.open('files/f1.png')
    figcol3.image(image3)

    ###
    st.session_state["pbar"] = st.session_state["pbar"] +10
    prog_bar.progress(st.session_state["pbar"])
    ###

    #########################
    # Figures - Feature Importance
    #########################
    fig_featimp_container.subheader("Feature Importance")

    image1 = Image.open('files/importances.png')
    fig_featimp_container.image(image1)
    
    ###
    st.session_state["pbar"] = st.session_state["pbar"] +30
    prog_bar.progress(st.session_state["pbar"])
    ###

