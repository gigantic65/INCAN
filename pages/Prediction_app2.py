import streamlit as st
import pandas as pd
import numpy as np
import sklearn.metrics as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import base64
import random
import itertools
from tensorflow.keras.models import load_model 

import PIL
from PIL import Image, ImageOps


 


def color_change(L,a,b):
    var_Y = ( L + 16 ) / 116
    var_X = a / 500 + var_Y
    var_Z = var_Y - b / 200
    
    if ( pow(var_Y,3) > 0.008856 ): var_Y = pow(var_Y,3) 
    else :                      var_Y = ( var_Y - 16 / 116 ) / 7.787
    if ( pow(var_X,3) > 0.008856 ): var_X = pow(var_X,3) 
    else :                      var_X = ( var_X - 16 / 116 ) / 7.787
    if ( pow(var_Z,3) > 0.008856 ): var_Z = pow(var_Z,3) 
    else :                      var_Z = ( var_Z - 16 / 116 ) / 7.787
    
    X = var_X * 95.047
    Y = var_Y * 100
    Z = var_Z * 108.883
    
    var_X = X / 100
    var_Y = Y / 100
    var_Z = Z / 100

    var_R = var_X *  3.2406 + var_Y * -1.5372 + var_Z * -0.4986
    var_G = var_X * -0.9689 + var_Y *  1.8758 + var_Z *  0.0415
    var_B = var_X *  0.0557 + var_Y * -0.2040 + var_Z *  1.0570
    
    if ( var_R > 0.0031308 ): var_R = 1.055 * ( pow(var_R,( 1 / 2.4 ))) - 0.055
    else :                    var_R = 12.92 * var_R
    if ( var_G > 0.0031308 ): var_G = 1.055 * ( pow(var_G,( 1 / 2.4 )) ) - 0.055
    else :                    var_G = 12.92 * var_G
    if ( var_B > 0.0031308 ): var_B = 1.055 * ( pow(var_B,( 1 / 2.4 ))) - 0.055
    else :                    var_B = 12.92 * var_B
    
    """if ( var_R > 0.004045 ): var_R = 1.055 * ( pow(var_R,( 1 / 2.4 ))) - 0.055
    else :                    var_R = 12.92 * var_R
    if ( var_G > 0.004045 ): var_G = 1.055 * ( pow(var_G,( 1 / 2.4 )) ) - 0.055
    else :                    var_G = 12.92 * var_G
    if ( var_B > 0.004045 ): var_B = 1.055 * ( pow(var_B,( 1 / 2.4 ))) - 0.055
    else :                    var_B = 12.92 * var_B"""
    
    
    sR = round(var_R * 255,0)
    sG = round(var_G * 255,0)
    sB = round(var_B * 255,0)
    
    return int(sR), int(sG), int(sB)

    

def app():
    
    """t_c0 = 1
    t_c1 = 1
    t_c2 = 1
    
    aa = pd.DataFrame()
    
    st.session_state[t_c0] = aa
    st.session_state[t_c1] = aa
    st.session_state[t_c2] = aa"""
    
    
    st.write('')
    st.write('')
    
    #st.markdown("<h6 style='text-align: right; color: black;'>?????? ??????: Incan UT6581, UT578A, UT578AF, UT578AS?????? </h6>", unsafe_allow_html=True)
    #st.markdown("<h6 style='text-align: right; color: black;'>??? ????????? ??????: 3740 Cases (Base A, B, C ??????)</h6>", unsafe_allow_html=True)

    st.write("")

    
    
    with st.expander("Predict New Conditions Guide"):
        st.write(
                "1. ????????? ?????? : Model accuracy ?????? ??????.\n"
                "2. ????????? ???????????? ?????? ?????? ????????? ?????? ????????? ?????? ????????? ??? ??????.\n"
        )


    
    #df1 = pd.read_csv('train.csv')
    model = pickle.load(open('./models_pcm.pkl', 'rb'))
    #scaler = pickle.load(open('./scaler_pcm.pkl', 'rb'))
    
    model1 = pickle.load(open('./models_pcm.pkl', 'rb'))
    model2 = pickle.load(open('./models_ET2.pkl', 'rb'))
    model3 = pickle.load(open('./models_ET3.pkl', 'rb'))
    model4 = pickle.load(open('./models_ET4.pkl', 'rb'))

    st.sidebar.write('')

    st.subheader('?????? ?????? ????????? ??????')
    st.write('')

    df = pd.read_csv('train_pcm.csv')

        
    x = list(df.columns[:-3])
    x2 = list(df.columns[:-3])
    y = list(df.columns[df.shape[1]-3:])

        #Selected_X = st.sidebar.multiselect('X variables', x, x)
        #Selected_y = st.sidebar.multiselect('Y variables', y, y)
            
    Selected_X = np.array(x)
    Selected_X2 = np.array(x2)
    Selected_y = np.array(y)
    
    
    col1,col2 = st.columns([1,1])

    
        
    with col1:    
        st.write('**X?????? ??? (????????? ????????? ???):**',Selected_X2.shape[0])
        st.info(list(Selected_X2))
        st.write('**?????? ?????? ?????? :**')
        st.info('Extra Tree Model')
        


    with col2:    
        st.write('**Y?????? ??? (?????????):**',Selected_y.shape[0])
        st.info(list(Selected_y))
        
        st.write('**???????????? ?????????(R2) :**')
        st.info('0.952')

    st.write('')   
            
    
    X = df[Selected_X]
    y = df[Selected_y]
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
    
    

    st.sidebar.write('')

    
    st.sidebar.header('???. ?????? ?????? ?????????')
    st.sidebar.write('')
    
    st.sidebar.write("<h5 style='text-align: left; color: black;'> 1. ?????? ?????? ?????? ?????????</h5>", unsafe_allow_html=True)

    ques = st.sidebar.radio('????????? ??????',('???????????? ????????????','???????????? ????????????', '???????????? ????????????'))
    
    #st.sidebar.write('**2.1 ?????? ?????? ?????? ?????? ??????**')
    #select = [1.5,0.5,1.0,2.0,5.0]
    #number = st.sidebar.selectbox("?????? ???????????? ???????????? : ", select)
                            
    st.sidebar.write('')
    st.sidebar.write('')
    st.sidebar.write('')
    st.sidebar.write('')
    st.sidebar.write('')
    st.sidebar.write('')

    st.write('')
    st.write('')
    st.write('')
    st.write('')
                             
    st.subheader('**???. ?????? ?????? ?????????**')
                
    Target_n = []
    Target_v = []
        
       
    if ques == '???????????? ????????????':
        

        st.write('')
        
        st.write("<h5 style='text-align: left; color: black;'> 1.1 ?????? ???????????? - ?????? ???????????? ????????????</h5>", unsafe_allow_html=True)
        st.markdown("""<hr style="height:2px;border:none; color:rgb(60,90,180); background-color:rgb(60,90,180);"/> """, unsafe_allow_html=True)

        #col1,col2 = st.columns([1,1])
        
        col3,col4,col5,col6 = st.columns([1,1,1,3])
        
        columns = ['Target_L','Target_a','Target_b']

        Target_n = []
        Target_v = []
            
            
        with col3:
            #st.write('')
            value1 = st.number_input(columns[0], -1000.00, 1000.00, 0.0,format="%.3f")
            Target_n.append(columns[0])
            Target_v.append(value1)
        with col4:
            #st.write('')
            value2 = st.number_input(columns[1], -1000.00, 1000.00, 0.0,format="%.3f")
            Target_n.append(columns[1])
            Target_v.append(value2)
        with col5:
            #st.write('')
            value3 = st.number_input(columns[2], -1000.00, 1000.00, 0.0,format="%.3f")
            Target_n.append(columns[2])
            Target_v.append(value3)
                
            
        with col3:
            st.write('')
            select = ['YJ2442']
            selected1 = st.selectbox("?????? ?????? : ", select)
            
        st.markdown("""<hr style="height:2px;border:none;color:rgb(60,90,180); background-color:rgb(60,90,180);" /> """, unsafe_allow_html=True)


        
            
        name2=[]
        test2=[]

        
        color_list =[]   
        color_list = ['YY1137W', 'YY1120K', 'YY1116Y', 'YY1111R', 'YY1110R', 'YY1112B', 'YY1113B', 'YY1104G', 'YY1103G', 'YY1169R']
        color_list = pd.DataFrame(color_list,columns=['color'])
        
        DT = pd.read_csv('train_pcm.csv')
    
        count = 0
        
        para3 = pd.DataFrame()
        
        
        #st.session_state[t_c0] = Target_v[0]
        #t_c0 = st.session_state[t_c0]
        #st.session_state[t_c1] = Target_v[1]
        #t_c1 = st.session_state[t_c1]
        #st.session_state[t_c2] = Target_v[2]
        #t_c2 = st.session_state[t_c2]
        

        #st.session_state[d_c] = para8_new
    
        if st.button('Run Prediction',key = count):
            

            
            
            if Target_v[0] < 45.0 and Target_v[1] >= 0.0 and Target_v[2] >= 0.0:
                model = model1
                para3 = pd.read_csv('pcmdata001.csv')                

                
            if Target_v[0] >= 45.0 and Target_v[0] < 65.0 and Target_v[1] >= 0.0 and Target_v[2] >= 0.0:
                model = model2
                para3 = pd.read_csv('pcmdata002.csv')

                
            if Target_v[0] >= 65.0 and Target_v[1] >= 0.0 and Target_v[2] >= 0.0:
                model = model1
                para3 = pd.read_csv('pcmdata003.csv')

                
            if Target_v[0] < 45.0 and Target_v[1] >= 0.0 and Target_v[2] < 0.0:
                model = model4
                para3 = pd.read_csv('pcmdata00_1.csv')


            if Target_v[0] >= 45.0 and Target_v[0] < 65.0 and Target_v[1] >= 0.0 and Target_v[2] < 0.0:
                model = model1
                para3 = pd.read_csv('pcmdata00_2.csv')


            if Target_v[0] >= 65.0 and Target_v[1] >= 0.0 and Target_v[2] < 0.0:
                model = model2
                para3 = pd.read_csv('pcmdata00_3.csv')
                
                

            if Target_v[0] < 45.0 and Target_v[1] < 0.0 and Target_v[2] >= 0.0:
                model = model1
                para3 = pd.read_csv('pcmdata0_01.csv')


            if Target_v[0] >= 45.0 and Target_v[0] < 65.0 and Target_v[1] < 0.0 and Target_v[2] >= 0.0:
                model = model3
                para3 = pd.read_csv('pcmdata0_02.csv')


            if Target_v[0] >= 65.0 and Target_v[1] < 0.0 and Target_v[2] >= 0.0:
                model = model2
                para3 = pd.read_csv('pcmdata0_03.csv')
       

            if Target_v[0] < 45.0 and Target_v[1] < 0.0 and Target_v[2] < 0.0:
                model = model1
                para3 = pd.read_csv('pcmdata0_0_1.csv')


            if Target_v[0] >= 45.0 and Target_v[0] < 65.0 and Target_v[1] < 0.0 and Target_v[2] < 0.0:
                model = model2
                para3 = pd.read_csv('pcmdata0_0_2.csv')


            if Target_v[0] >= 65.0 and Target_v[1] < 0.0 and Target_v[2] < 0.0:
                model = model3
                para3 = pd.read_csv('pcmdata0_0_3.csv')




            
            

            with col3:

                st.write('')
                st.write('')
                r, g, b = color_change(Target_v[0] ,Target_v[1] , Target_v[2])
                
                img = Image.new("RGB", (250, 50), color=(r,g,b))
                st.image(img, caption='Target')
                #st.write(t_c0,t_c1,t_c2)
                
            
            with col6:
                
                
                st.markdown("<h6 style='text-align: left; color: darkblue;'> 1.1. ???????????? ???????????? </h6>", unsafe_allow_html=True)
            
            

                Target_v = pd.DataFrame(Target_v)
                Target_v = Target_v.T
                Target_v.columns = Target_n
                
                
                DT = DT.reset_index(drop=True)
                
                DT['Delta_E'] = 0.0
                
                
                for i in range(DT.shape[0]):
                    DT['Delta_E'][i] = ((Target_v['Target_L'].values - DT['Target_L'][i])**2+(Target_v['Target_a'].values - DT['Target_a'][i])**2+(Target_v['Target_b'].values - DT['Target_b'][i])**2)**0.5 
                        
                DT.sort_values(by='Delta_E', ascending=True, inplace =True)
                
                
                
                
                DT = DT.head(3)
                
                DT = DT.reset_index(drop=True)
                
                
                
                st.markdown("<h6 style='text-align: left; color: darkblue;'> ??? ?????? ?????? ?????? ?????? </h6>", unsafe_allow_html=True)
    
                #st.write(DT)
                
                DT_new = DT.replace(0,np.nan)
                DT_new = pd.DataFrame(DT_new)
                DT_new = DT_new.dropna(how='all',axis=1)
                DT_new = DT_new.replace(np.nan,0)
                st.write(DT_new)
                
                
                
                
                
                
                
                DT2 = DT.drop(['Target_L','Target_a','Target_b','Delta_E'], axis=1)
                
                datafile = DT2.values
                
                #rescaleddatafile = scaler.transform(datafile)
    
                predictions2 = model.predict(datafile)
                
                predictions2 = pd.DataFrame(predictions2, columns=['Pred_L','Pred_a','Pred_b'])
                    
                para4 = pd.concat([DT2,predictions2], axis=1)
                
                para4['Delta_E'] = DT['Delta_E']
                   
                para4 = para4.reset_index(drop=True)
                
                
                #st.write('check')
                #st.write(para4['Pred_L'])
                #st.write(para4['Pred_L'].mean())
                
                #st.write(para4)
                #st.write(DT)
                
                
                #st.write(Diff)
                
                
                
                # ???????????? ????????? ????????? ????????? Target??? ???????????? ????????? Target?????? ?????????.
                NTarget_v1 = []
                NTarget_v1.append(Target_v['Target_L']+(para4['Pred_L'][0]-DT['Target_L'][0]))
                NTarget_v1.append(Target_v['Target_a']+(para4['Pred_a'][0]-DT['Target_a'][0]))
                NTarget_v1.append(Target_v['Target_b']+(para4['Pred_b'][0]-DT['Target_b'][0]))
                
                NTarget_v1 = pd.DataFrame(NTarget_v1)
                NTarget_v1 = NTarget_v1.T
                NTarget_v1.columns = Target_n
                
                Diff1 = Target_v - NTarget_v1
     
                #st.write(NTarget_v1)
    
    
                NTarget_v2 = []
                NTarget_v2.append(Target_v['Target_L']+(para4['Pred_L'][1]-DT['Target_L'][1]))
                NTarget_v2.append(Target_v['Target_a']+(para4['Pred_a'][1]-DT['Target_a'][1]))
                NTarget_v2.append(Target_v['Target_b']+(para4['Pred_b'][1]-DT['Target_b'][1]))
                
                NTarget_v2 = pd.DataFrame(NTarget_v2)
                NTarget_v2 = NTarget_v2.T
                NTarget_v2.columns = Target_n
                
                Diff2 = Target_v - NTarget_v2
                
                #st.write(NTarget_v2)
    
                
                
                NTarget_v3 = []
                NTarget_v3.append(Target_v['Target_L']+(para4['Pred_L'][2]-DT['Target_L'][2]))
                NTarget_v3.append(Target_v['Target_a']+(para4['Pred_a'][2]-DT['Target_a'][2]))
                NTarget_v3.append(Target_v['Target_b']+(para4['Pred_b'][2]-DT['Target_b'][2]))
                
                
                NTarget_v3 = pd.DataFrame(NTarget_v3)
                NTarget_v3 = NTarget_v3.T
                NTarget_v3.columns = Target_n
                
                Diff3 = Target_v - NTarget_v3
                
                
                #st.write(NTarget_v3)
    
                
    
                
                #st.write('target=', Target_v)
                #st.write('actual=', DT)
                #st.write('prediction=', para4)
                
                #st.write('new_target=', NTarget_v1)
                
                #st.write(Diff)
                
                #st.write(NTarget_v)
                
                # ????????? ?????? ???????????? ????????? ????????????, ?????? ????????? ?????? ??????????????? ??????????????? ??????.
                para55 = para4
                para55['Pred_L'] = DT['Target_L']
                para55['Pred_a'] = DT['Target_a']
                para55['Pred_b'] = DT['Target_b']
                
                df_min0 = DT.iloc[0]
                df_min1 = DT.iloc[1]
                df_min2 = DT.iloc[2]
                
                #df_min0 = pd.DataFrame(df_min0)
                #test = df_min0[df_min0 !=0]
                #st.write(test)
                #st.write(df_min0)
                #st.write(df_min0.shape)
        
    
                #st.write(df_min0)
                #st.write(df_min0.shape())
                

                # 3?????? ???????????? ?????? +- ????????? ????????????, ??????????????? ????????? ?????????.
                para2 =[]
                para6 = pd.DataFrame()
                
                col_list = []
                for j in range(df_min0.shape[0]-4):
                        
                    column = df_min0.index[j]
                    

                    if df_min0.iloc[j] > 0 and df_min0.iloc[j] < 20:
                        
                        min = round(df_min0.iloc[j] - 5,2)
                        if min <0: min = 0 
                        max = round(df_min0.iloc[j] + 5,2)
                        #st.write(max, min)
                        para = np.arange(min, max, (max-min)/50.0)  
                        col_list.append(column)
                        para2.append(para)
                        
                    if df_min0.iloc[j] >= 20 and df_min0.iloc[j] < 40:
                        
                        min = round(df_min0.iloc[j] - 10,2)
                        if min <0: min = 0 
                        max = round(df_min0.iloc[j] + 10,2)
                        #st.write(max, min)
                        para = np.arange(min, max, (max-min)/50.0)  
                        col_list.append(column)
                        para2.append(para)
                        
                    if df_min0.iloc[j] >= 40:
                        min = round(df_min0.iloc[j] - 20,2)
                        if min <0: min = 0 
                        max = round(df_min0.iloc[j] + 20,2)
                        #st.write(max, min)
                        para = np.arange(min, max, (max-min)/50.0)  
                        col_list.append(column)
                        para2.append(para)
                                              
                para2 = pd.DataFrame(para2)
                para2 = para2.T
                #st.write(col_list)
                para6 = para2
                para6.columns = col_list
                
                #st.write(para6)
        
        
                    
                
                para21 =[]
                para61 = pd.DataFrame()
        
                col_list1 = []
                for j in range(df_min1.shape[0]-4):
                        
                    column = df_min1.index[j]
        
                    if df_min1.iloc[j] > 0 and df_min1.iloc[j] < 20:
                        min = round(df_min1.iloc[j] - 5,2)
                        if min <0: min = 0 
                        max = round(df_min1.iloc[j] + 5,2)
                        #st.write(max, min)
                        para = np.arange(min, max, (max-min)/50.0)  
                        col_list1.append(column)
                        para21.append(para)
                        
                        
                    if df_min1.iloc[j] >= 20 and df_min1.iloc[j] < 40:
                        min = round(df_min1.iloc[j] - 10,2)
                        if min <0: min = 0 
                        max = round(df_min1.iloc[j] + 10,2)
                        #st.write(max, min)
                        para = np.arange(min, max, (max-min)/50.0)  
                        col_list1.append(column)
                        para21.append(para)
                        
                        
                    if df_min1.iloc[j] >= 40:
                        min = round(df_min1.iloc[j] - 20,2)
                        if min <0: min = 0 
                        max = round(df_min1.iloc[j] + 20,2)
                        #st.write(max, min)
                        para = np.arange(min, max, (max-min)/50.0)  
                        col_list1.append(column)
                        para21.append(para)
                            
                para21 = pd.DataFrame(para21)
                para21 = para21.T
                #st.write(col_list1)
                para61 = para21
                para61.columns = col_list1
                #st.write(para61)
        
                
                
                para22 =[]
                para62 = pd.DataFrame()
                col_list2 = []
                for j in range(df_min2.shape[0]-4):
                        
                    column = df_min2.index[j]
                    
                    if df_min2.iloc[j] > 0 and df_min2.iloc[j] < 20:
                        min = round(df_min2.iloc[j] - 5,2)
                        if min <0: min = 0 
                        max = round(df_min2.iloc[j] + 5,2)
                        #st.write(max, min)
                        para = np.arange(min, max, (max-min)/50.0)  
                        col_list2.append(column)
                        para22.append(para)
                        
                        
                    if df_min2.iloc[j] >= 20 and df_min2.iloc[j] < 40:
                        min = round(df_min2.iloc[j] - 10,2)
                        if min <0: min = 0 
                        max = round(df_min2.iloc[j] + 10,2)
                        #st.write(max, min)
                        para = np.arange(min, max, (max-min)/50.0)  
                        col_list2.append(column)
                        para22.append(para)    

                        
                    if df_min2.iloc[j] >= 40:
                        min = round(df_min2.iloc[j] - 20,2)
                        if min <0: min = 0 
                        max = round(df_min2.iloc[j] + 20,2)
                        #st.write(max, min)
                        para = np.arange(min, max, (max-min)/50.0)  
                        col_list2.append(column)
                        para22.append(para)
                            
                para22 = pd.DataFrame(para22)
                para22 = para22.T
                #st.write(col_list1)
                para62 = para22
                para62.columns = col_list2
                #st.write(para62)
        
                
                
                    
                    
                
                
                New_x2 = pd.DataFrame(X.iloc[0,:])
                New_x2 = New_x2.T
                
                random.seed(42)
                para7 = []
                for i in range(5000):
                    para5 = []
                    for col in New_x2.columns:
                        New_x2[col] = 0.0
                                           
                        for col1 in list(para6.columns):
                            if col1 == col:
                                New_x2[col] = random.sample(list(para6[col1]),1)
                                    
                                                          
                        para5.append(float(New_x2[col].values))
                      
                    para7.append(para5)
                           
        
                para7 = pd.DataFrame(para7, columns=X.columns) 
                para7 = para7.replace(np.nan, 0)
                
                
                para71 = []
                random.seed(42)
                for i in range(5000):
                    para5 = []
                    for col in New_x2.columns:
                        New_x2[col] = 0.0
                                           
                        for col1 in list(para61.columns):
                            if col1 == col:
                                New_x2[col] = random.sample(list(para61[col1]),1)
                                    
                                                         
                        para5.append(float(New_x2[col].values))
                      
                    para71.append(para5)
                           
        
                para71 = pd.DataFrame(para71, columns=X.columns) 
                para71 = para71.replace(np.nan, 0)
                
                
                
                para72 = []
                random.seed(42)
                for i in range(500):
                    para5 = []
                    for col in New_x2.columns:
                        New_x2[col] = 0.0
                                           
                        for col1 in list(para62.columns):
                            if col1 == col:
                                New_x2[col] = random.sample(list(para62[col1]),1)
                                    
                                                         
                        para5.append(float(New_x2[col].values))
                      
                    para72.append(para5)
                           
                para72 = pd.DataFrame(para72, columns=X.columns)
                para72 = para72.replace(np.nan, 0)
                
        
        
                
                #para7 = para7.drop_duplicates()
                #para7 = para7.reset_index(drop=True)
                
                # ???????????? ???????????? ?????? ????????? ????????????, ????????? ???????????? ?????? ?????? ?????? ?????? ?????????.
        
                datafile2 = para7.values
                
                #st.write(para7)
                
                #rescaleddatafile2 = scaler.transform(datafile2)
                   
                predictions3 = model.predict(datafile2)
           
                predictions3 = pd.DataFrame(predictions3, columns=['Pred_L','Pred_a','Pred_b'])
                   
                para7 = pd.concat([para7,predictions3], axis=1)
                       
                para7 = para7.reset_index(drop = True)
                
                para7['Delta_E'] = 0.0
                                  
                for i in range(para7.shape[0]):
        
                    para7['Delta_E'][i] = ((NTarget_v1['Target_L'] - predictions3['Pred_L'][i])**2+(NTarget_v1['Target_a'] - predictions3['Pred_a'][i])**2+(NTarget_v1['Target_b'] - predictions3['Pred_b'][i])**2)**0.5 
                
                para7.sort_values(by='Delta_E', ascending=True, inplace =True)
                para7 = para7.head(1)
                para77 = para55[:1]
                #st.write(para7)
                #st.write(para77)
                
                para7 = pd.concat([para7,para77], axis=0)
                para7.sort_values(by='Delta_E', ascending=True, inplace =True)
                
                para7 = para7.head(1)
        
                #st.write(para7)
                
                
                
                
                datafile2 = para71.values
                
                #st.write(datafile2)
                #rescaleddatafile2 = scaler.transform(datafile2)
                   
                predictions3 = model.predict(datafile2)
           
                predictions3 = pd.DataFrame(predictions3, columns=['Pred_L','Pred_a','Pred_b'])
                   
                para71 = pd.concat([para71,predictions3], axis=1)
                       
                para71 = para71.reset_index(drop = True)
                
                para71['Delta_E'] = 0.0
                                  
                for i in range(para71.shape[0]):
        
                    para71['Delta_E'][i] = ((NTarget_v2['Target_L'] - predictions3['Pred_L'][i])**2+(NTarget_v2['Target_a'] - predictions3['Pred_a'][i])**2+(NTarget_v2['Target_b'] - predictions3['Pred_b'][i])**2)**0.5 
                
                para71.sort_values(by='Delta_E', ascending=True, inplace =True)
                
                para71 = para71.head(1)
                para717 = para55[1:2]
                
                
                para71 = pd.concat([para71,para717], axis=0)
                para71.sort_values(by='Delta_E', ascending=True, inplace =True)
                
                para71 = para71.head(1)
                #st.write(para71)
                
                
                
                
                datafile2 = para72.values
                
                #rescaleddatafile2 = scaler.transform(datafile2)
                   
                predictions3 = model.predict(datafile2)
           
                predictions3 = pd.DataFrame(predictions3, columns=['Pred_L','Pred_a','Pred_b'])
                   
                para72 = pd.concat([para72,predictions3], axis=1)
                       
                para72 = para72.reset_index(drop = True)
                
                para72['Delta_E'] = 0.0
                                  
                for i in range(para72.shape[0]):
        
                    para72['Delta_E'][i] = ((NTarget_v3['Target_L'] - predictions3['Pred_L'][i])**2+(NTarget_v3['Target_a'] - predictions3['Pred_a'][i])**2+(NTarget_v3['Target_b'] - predictions3['Pred_b'][i])**2)**0.5 
                
                para72.sort_values(by='Delta_E', ascending=True, inplace =True)
                
                para72 = para72.head(1)
                para727 = para55[2:3]
                
                para72 = pd.concat([para72,para727], axis=0)
                para72.sort_values(by='Delta_E', ascending=True, inplace =True)
                
                para72 = para72.head(1)
                #st.write(para72)
                
                
                
                #st.write('**2??? ?????? ????????? ??????:**')
                
                para7 = pd.concat([para7,para71,para72], axis=0)
                
                #para7.sort_values(by='Delta_E', ascending=True, inplace =True)
                
                para7 = para7.reset_index(drop=True)
                st.write('')
                
                #st.markdown("<h6 style='text-align: left; color: darkblue;'> 1??? ?????? ?????? ?????? ?????? </h6>", unsafe_allow_html=True)
                
                #st.write(para7)
                
                
                """st.write('**2??? ?????? ????????? ??????:**')
                
                df_min0 = para4.head(3)
                df_min1 = para7.head(3)
                
                df_min11 = pd.concat([df_min0,df_min1], axis=0)
                
                df_min11.sort_values(by='Delta_E', ascending=True, inplace =True)
                
                
                
                st.write(para7)"""
                
                
                
                
                df_min3 = para7.iloc[0]
                df_min4 = para7.iloc[1]
                df_min5 = para7.iloc[2]
                
                
        
                para23 =[]
                para63 = pd.DataFrame()
        
                col_list = []
                for j in range(df_min3.shape[0]-4):
                        
                    column = df_min3.index[j]
                    
                    if df_min3.iloc[j] > 0 and df_min3.iloc[j] < 20:
                        min = round(df_min3.iloc[j] - 2,3)
                        if min <0: min = 0 
                        max = round(df_min3.iloc[j] + 2,3)
                        #st.write(max, min)
                        para = np.arange(min, max, (max-min)/50.0)  
                        col_list.append(column)
                        para23.append(para)
                        
                    if df_min3.iloc[j] >= 20:
                        min = round(df_min3.iloc[j] - 4,3)
                        if min <0: min = 0 
                        max = round(df_min3.iloc[j] + 4,3)
                        #st.write(max, min)
                        para = np.arange(min, max, (max-min)/50.0)  
                        col_list.append(column)
                        para23.append(para)
                                              
                para23 = pd.DataFrame(para23)
                para23 = para23.T
                #st.write(col_list)
                para63 = para23
                para63.columns = col_list
                
                
                
                para24 =[]
                para64 = pd.DataFrame()
        
                col_list = []
                for j in range(df_min4.shape[0]-4):
                        
                    column = df_min4.index[j]
                    
                        
                    if df_min4.iloc[j] > 0 and df_min4.iloc[j] < 20:
                        min = round(df_min4.iloc[j] - 2,3)
                        if min <0: min = 0 
                        max = round(df_min4.iloc[j] + 2,3)
                        #st.write(max, min)
                        para = np.arange(min, max, (max-min)/50.0)  
                        col_list.append(column)
                        para24.append(para)
                        
                    if df_min4.iloc[j] >= 20:
                        min = round(df_min4.iloc[j] - 4,3)
                        if min <0: min = 0 
                        max = round(df_min4.iloc[j] + 4,3)
                        #st.write(max, min)
                        para = np.arange(min, max, (max-min)/50.0)  
                        col_list.append(column)
                        para24.append(para)
                                              
                para24 = pd.DataFrame(para24)
                para24 = para24.T
                #st.write(col_list)
                para64 = para24
                para64.columns = col_list
                
                
                
                para25 =[]
                para65 = pd.DataFrame()
        
                col_list = []
                for j in range(df_min5.shape[0]-4):
                        
                    column = df_min5.index[j]
                    
                        
                    if df_min5.iloc[j] > 0 and df_min5.iloc[j] < 20:
                        min = round(df_min5.iloc[j] - 2,3)
                        if min <0: min = 0 
                        max = round(df_min5.iloc[j] + 2,3)
                        #st.write(max, min)
                        para = np.arange(min, max, (max-min)/50.0)  
                        col_list.append(column)
                        para25.append(para)
                        
                    if df_min5.iloc[j] >= 20:
                        min = round(df_min5.iloc[j] - 4,3)
                        if min <0: min = 0 
                        max = round(df_min5.iloc[j] + 4,3)
                        #st.write(max, min)
                        para = np.arange(min, max, (max-min)/50.0)  
                        col_list.append(column)
                        para25.append(para)
                                              
                para25 = pd.DataFrame(para25)
                para25 = para25.T
                #st.write(col_list)
                para65 = para25
                para65.columns = col_list
                
         
        
                    
                #st.write(para61)
                
                New_x2 = pd.DataFrame(X.iloc[0,:])
                New_x2 = New_x2.T
                
                para8 = []
                random.seed(42)
                for i in range(5000):
                    para5 = []
                    for col in New_x2.columns:
                        New_x2[col] = 0.0
                                           
                        for col1 in list(para63.columns):
                            if col1 == col:
                                New_x2[col] = random.sample(list(para63[col1]),1)
                                    
                                                          
                        para5.append(float(New_x2[col].values))
                      
                    para8.append(para5)
                           
        
                para8 = pd.DataFrame(para8, columns=X.columns) 
                
                para8 = para8.replace(np.nan,0)
                
                
                para81 = []
                random.seed(42)
                for i in range(5000):
                    para5 = []
                    for col in New_x2.columns:
                        New_x2[col] = 0.0
                                           
                        for col1 in list(para64.columns):
                            if col1 == col:
                                New_x2[col] = random.sample(list(para64[col1]),1)
                                    
                                                        
                        para5.append(float(New_x2[col].values))
                      
                    para81.append(para5)
                           
        
                para81 = pd.DataFrame(para81, columns=X.columns) 
                para81 = para81.replace(np.nan,0)
                
                
                para82 = []
                random.seed(42)
                for i in range(500):
                    para5 = []
                    for col in New_x2.columns:
                        New_x2[col] = 0.0
                                           
                        for col1 in list(para65.columns):
                            if col1 == col:
                                New_x2[col] = random.sample(list(para65[col1]),1)
                                    
                                                         
                        para5.append(float(New_x2[col].values))
                      
                    para82.append(para5)
                           
        
                para82 = pd.DataFrame(para82, columns=X.columns) 
                para82 = para82.replace(np.nan,0)
                
        
        
                datafile3 = para8.values
        
                #rescaleddatafile3 = scaler.transform(datafile3)
                   
                predictions4 = model.predict(datafile3)
           
                predictions4 = pd.DataFrame(predictions4, columns=['Pred_L','Pred_a','Pred_b'])
                   
                para8 = pd.concat([para8,predictions4], axis=1)
                   
                para8 = para8.reset_index(drop = True)
                
                para8['Delta_E'] = 0.0
                                  
                for i in range(para8.shape[0]):
        
                    para8['Delta_E'][i] = ((NTarget_v1['Target_L'] - predictions4['Pred_L'][i])**2+(NTarget_v1['Target_a'] - predictions4['Pred_a'][i])**2+(NTarget_v1['Target_b'] - predictions4['Pred_b'][i])**2)**0.5 
                
                para8.sort_values(by='Delta_E', ascending=True, inplace =True)
                
                para8 = para8.head(1)
                para88 = para7[:1]
                
                
                para8 = pd.concat([para8,para88], axis=0)
                para8.sort_values(by='Delta_E', ascending=True, inplace =True)
                
                para8 = para8.head(1)
        
                #st.write(para8)
                
                
                
                datafile3 = para81.values
        
                #rescaleddatafile3 = scaler.transform(datafile3)
                   
                predictions4 = model.predict(datafile3)
           
                predictions4 = pd.DataFrame(predictions4, columns=['Pred_L','Pred_a','Pred_b'])
                   
                para81 = pd.concat([para81,predictions4], axis=1)
                   
                para81 = para81.reset_index(drop = True)
                
                para81['Delta_E'] = 0.0
                                  
                for i in range(para81.shape[0]):
        
                    para81['Delta_E'][i] = ((NTarget_v2['Target_L'] - predictions4['Pred_L'][i])**2+(NTarget_v2['Target_a'] - predictions4['Pred_a'][i])**2+(NTarget_v2['Target_b'] - predictions4['Pred_b'][i])**2)**0.5 
                
                para81.sort_values(by='Delta_E', ascending=True, inplace =True)
                
                para81 = para81.head(1)
                para818 = para7[1:2]
                
                
                para81 = pd.concat([para81,para818], axis=0)
                para81.sort_values(by='Delta_E', ascending=True, inplace =True)
                
                para81 = para81.head(1)
        
                #st.write(para81)
                
                
                
                datafile3 = para82.values
        
                #rescaleddatafile3 = scaler.transform(datafile3)
                   
                predictions4 = model.predict(datafile3)
           
                predictions4 = pd.DataFrame(predictions4, columns=['Pred_L','Pred_a','Pred_b'])
                   
                para82 = pd.concat([para82,predictions4], axis=1)
                   
                para82 = para82.reset_index(drop = True)
                
                para82['Delta_E'] = 0.0
                                  
                for i in range(para82.shape[0]):
        
                    para82['Delta_E'][i] = ((NTarget_v3['Target_L'] - predictions4['Pred_L'][i])**2+(NTarget_v3['Target_a'] - predictions4['Pred_a'][i])**2+(NTarget_v3['Target_b'] - predictions4['Pred_b'][i])**2)**0.5 
                
                para82.sort_values(by='Delta_E', ascending=True, inplace =True)
                
                para82 = para82.head(1)
                para828 = para7[2:3]
                
                
                para82 = pd.concat([para82,para828], axis=0)
                para82.sort_values(by='Delta_E', ascending=True, inplace =True)
                
                para82 = para82.head(1)
        
                #st.write(para82)
                
                
                st.markdown("<h6 style='text-align: left; color: darkblue;'> ???  ?????? ?????? ?????? ?????? ?????? </h6>", unsafe_allow_html=True)
                            
                para8 = pd.concat([para8,para81,para82], axis=0)
                
                #st.write(para8)
                
                # ?????? ?????????????????? ?????????????????? ????????? (??????????????? ???)
                #st.write(Diff1,Diff2,Diff3)
                
                para8 = para8.reset_index(drop=True)    
                
                para8['Pred_L'][0] = para8['Pred_L'][0] + Diff1['Target_L']
                para8['Pred_a'][0] = para8['Pred_a'][0] + Diff1['Target_a']
                para8['Pred_b'][0] = para8['Pred_b'][0] + Diff1['Target_b']
                
                para8['Pred_L'][1] = para8['Pred_L'][1] + Diff2['Target_L']
                para8['Pred_a'][1] = para8['Pred_a'][1] + Diff2['Target_a']
                para8['Pred_b'][1] = para8['Pred_b'][1] + Diff2['Target_b']
                
                para8['Pred_L'][2] = para8['Pred_L'][2] + Diff3['Target_L']
                para8['Pred_a'][2] = para8['Pred_a'][2] + Diff3['Target_a']
                para8['Pred_b'][2] = para8['Pred_b'][2] + Diff3['Target_b']
    
    
    
                para8.sort_values(by='Delta_E', ascending=True, inplace =True)
                
                para8 = para8.reset_index(drop=True)     
                
                
                
                r, g, b = color_change(para8['Pred_L'][0] ,para8['Pred_a'][0] , para8['Pred_b'][0])
                img = Image.new("RGB", (250, 50), color=(r,g,b))
                st.image(img, caption='Prediction')
                
                
                
    
                para8['Delta_L'] = 0.0
                para8['Delta_a'] = 0.0
                para8['Delta_b'] = 0.0
                
                for i in range(para8.shape[0]):
                    para8['Delta_L'][i] = (Target_v['Target_L'] - para8['Pred_L'][i]) 
                    para8['Delta_a'][i] = (Target_v['Target_a'] - para8['Pred_a'][i])
                    para8['Delta_b'][i] = (Target_v['Target_b'] - para8['Pred_b'][i])
                    
                #st.write(para8)
                    
                para8 = para8.drop(['Pred_L','Pred_a','Pred_b'], axis=1)
                
                
                
                
                para8_new = para8.replace(0,np.nan)
                para8_new = pd.DataFrame(para8_new)
                para8_new = para8_new.dropna(how='all',axis=1)
                para8_new = para8_new.replace(np.nan,0)
                
                
                para8_new2 = para8_new.drop(['Delta_E'], axis=1)
                para8_new2['Delta_E'] = para8_new['Delta_E']
                para8_new = para8_new2
                
                
                
                
                
                
                if para8['Delta_E'][0] < 0.5:
                    st.markdown("<h6 style='text-align: left; color: black;'> ?????? ?????? ????????? 0.5 ?????? ?????? ??????????????? ???????????? ???????????? ?????? ??????. </h6>", unsafe_allow_html=True)
                    
                else:
                    st.markdown("<h6 style='text-align: left; color: black;'> ?????? ?????? ????????? 0.5 ?????? ?????? ??????????????? ???????????? ???????????? ?????? ??????. </h6>", unsafe_allow_html=True)
                
                

                


                st.write(para8_new)
                
                
                """ if st.button('????????????'):
                    st.session_state[t_c] = Target_v
                    st.session_state[d_c] = para8_new"""
                
                





    if ques == '???????????? ????????????':
        

        st.write('')

        st.write("<h5 style='text-align: left; color: black;'> 1.2 ?????? ???????????? - ???????????? ????????????</h5>", unsafe_allow_html=True)
        st.markdown("""<hr style="height:2px;border:none; color:rgb(60,90,180); background-color:rgb(60,90,180);"/> """, unsafe_allow_html=True)


        #col1,col2 = st.columns([1,1])
        
        col3,col4,col5,col6 = st.columns([1,1,1,3])
        
        columns = ['Target_L','Target_a','Target_b']

        Target_n = []
        Target_v = []
            
            
        with col3:
            value1 = st.number_input(columns[0], -1000.00, 1000.00, 0.0,format="%.3f")
            Target_n.append(columns[0])
            Target_v.append(value1)
        with col4:
            value2 = st.number_input(columns[1], -1000.00, 1000.00, 0.0,format="%.3f")
            Target_n.append(columns[1])
            Target_v.append(value2)
        with col5:
            value3 = st.number_input(columns[2], -1000.00, 1000.00, 0.0,format="%.3f")
            Target_n.append(columns[2])
            Target_v.append(value3)
                
            
        with col3:
            st.write('')
            select = ['YJ2442']
            selected1 = st.selectbox("?????? ?????? : ", select)
            
        st.markdown("""<hr style="height:2px;border:none; color:rgb(60,90,180); background-color:rgb(60,90,180);"/> """, unsafe_allow_html=True)        


        
            
        name2=[]
        test2=[]

        
        color_list =[]   
        color_list = ['YY1137W', 'YY1120K', 'YY1116Y', 'YY1111R', 'YY1110R', 'YY1112B', 'YY1113B', 'YY1104G', 'YY1103G', 'YY1169R']
        color_list = pd.DataFrame(color_list,columns=['color'])
        
        DT = pd.read_csv('train_pcm.csv')
    
        count = 0
        
        para3 = pd.DataFrame()
        
    
        if st.button('Run Prediction',key = count):
            
            if Target_v[0] < 45.0 and Target_v[1] >= 0.0 and Target_v[2] >= 0.0:
                model = model1
                para3 = pd.read_csv('pcmdata001.csv')                

                
            if Target_v[0] >= 45.0 and Target_v[0] < 65.0 and Target_v[1] >= 0.0 and Target_v[2] >= 0.0:
                model = model2
                para3 = pd.read_csv('pcmdata002.csv')

                
            if Target_v[0] >= 65.0 and Target_v[1] >= 0.0 and Target_v[2] >= 0.0:
                model = model1
                para3 = pd.read_csv('pcmdata003.csv')

                
            if Target_v[0] < 45.0 and Target_v[1] >= 0.0 and Target_v[2] < 0.0:
                model = model4
                para3 = pd.read_csv('pcmdata00_1.csv')


            if Target_v[0] >= 45.0 and Target_v[0] < 65.0 and Target_v[1] >= 0.0 and Target_v[2] < 0.0:
                model = model1
                para3 = pd.read_csv('pcmdata00_2.csv')


            if Target_v[0] >= 65.0 and Target_v[1] >= 0.0 and Target_v[2] < 0.0:
                model = model2
                para3 = pd.read_csv('pcmdata00_3.csv')
                
                

            if Target_v[0] < 45.0 and Target_v[1] < 0.0 and Target_v[2] >= 0.0:
                model = model1
                para3 = pd.read_csv('pcmdata0_01.csv')


            if Target_v[0] >= 45.0 and Target_v[0] < 65.0 and Target_v[1] < 0.0 and Target_v[2] >= 0.0:
                model = model3
                para3 = pd.read_csv('pcmdata0_02.csv')


            if Target_v[0] >= 65.0 and Target_v[1] < 0.0 and Target_v[2] >= 0.0:
                model = model2
                para3 = pd.read_csv('pcmdata0_03.csv')
       

            if Target_v[0] < 45.0 and Target_v[1] < 0.0 and Target_v[2] < 0.0:
                model = model1
                para3 = pd.read_csv('pcmdata0_0_1.csv')


            if Target_v[0] >= 45.0 and Target_v[0] < 65.0 and Target_v[1] < 0.0 and Target_v[2] < 0.0:
                model = model2
                para3 = pd.read_csv('pcmdata0_0_2.csv')


            if Target_v[0] >= 65.0 and Target_v[1] < 0.0 and Target_v[2] < 0.0:
                model = model3
                para3 = pd.read_csv('pcmdata0_0_3.csv')
                
                
            

            with col3:

                st.write('')
                st.write('')
                r, g, b = color_change(Target_v[0] ,Target_v[1] , Target_v[2])
                
                img = Image.new("RGB", (250, 50), color=(r,g,b))
                st.image(img, caption='Target')
                
            
            with col6:
                
    
                st.markdown("<h6 style='text-align: left; color: darkblue;'> 1.2. ???????????? ???????????? </h6>", unsafe_allow_html=True)
                                   
                    
                st.markdown("<h6 style='text-align: left; color: darkblue;'> ****?????? ?????? ?????? ?????? ???</h6>", unsafe_allow_html=True)
                
                                       
                
                #st.write(para3)
                
                para3 = para3.drop(['sum'], axis=1)
    
                datafile = para3.values
                
                
                #rescaleddatafile = scaler.transform(datafile)
                   
                   
                predictions2 = model.predict(datafile)
           
                predictions2 = pd.DataFrame(predictions2, columns=['Pred_L','Pred_a','Pred_b'])
                   
                    
                para4 = pd.concat([para3,predictions2], axis=1)
    
                   
                para4 = para4.reset_index(drop=True)
                y = y.reset_index(drop=True)
                
                
                Target_v = pd.DataFrame(Target_v)
                Target_v = Target_v.T
                Target_v.columns = Target_n
    
    
                para4['Delta_E'] = 0.0
                
                for i in range(para4.shape[0]):
                    para4['Delta_E'][i] = ((Target_v['Target_L'] - predictions2['Pred_L'][i])**2+(Target_v['Target_a'] - predictions2['Pred_a'][i])**2+(Target_v['Target_b'] - predictions2['Pred_b'][i])**2)**0.5 
    
                para4.sort_values(by='Delta_E', ascending=True, inplace =True)
                
                #check = para4[para4['Delta_E']<1.5].count()
                
                #st.write(check[0])
    
    
        
    
                
                
                #st.markdown("<h6 style='text-align: left; color: darkblue;'> 3. ?????? ?????? ?????? ?????? </h6>", unsafe_allow_html=True)
                
                #st.write(para4)
                       
                df_min = para4.head(20)
                
                #st.write(df_min)
                
                
                # ????????? ????????? ?????? ??? ????????? ???????????? ??????
                
                check = []
                for i in range(df_min.shape[0]):
                    
                    test = df_min.iloc[i]
                    test = test[test !=0]
                                  
                    check.append(test.index)
                    
                    
                check = pd.DataFrame(check)  
                check = check.drop_duplicates()
                
                list1 = list(check.index)
                #st.write(list1)
                new_df = []
                for i in list1:
                    #st.write(i)
                    new_df.append(df_min.iloc[i])
                    
                #st.write(check)
                
                new_df = pd.DataFrame(new_df)
                #st.write(new_df)
                
                new_df = new_df.reset_index(drop=True)
                st.write('')
        
                st.markdown("<h6 style='text-align: left; color: darkblue;'> ??? 1??? ?????? ?????? ?????? </h6>", unsafe_allow_html=True)
                #st.write('**1??? ?????? ????????? ??????:**')
                #st.write(new_df)
                
                
                new_df_new = new_df.replace(0,np.nan)
                new_df_new = pd.DataFrame(new_df_new)
                new_df_new = new_df_new.dropna(how='all',axis=1)
                new_df_new = new_df_new.replace(np.nan,0)
                st.write(new_df_new)
                
                
                
                
                
                #st.write(new_df.shape[0])
                
                if new_df.shape[0]==1:
                            
                    st.write('')
                    st.write('')
                    
                    df_min0 = df_min.iloc[0]
                    
                    
                    #df_min0 = pd.DataFrame(df_min0)
                    #test = df_min0[df_min0 !=0]
                    #st.write(test)
                    #st.write(df_min0)
                    #st.write(df_min0.shape)
            
                    
                    
                    para2 =[]
                    para6 = pd.DataFrame()
            
                    col_list = []
                    for j in range(df_min0.shape[0]-4):
                            
                        column = df_min0.index[j]
                        
             
                    if df_min0.iloc[j] > 0 and df_min0.iloc[j] < 20:
                        min = round(df_min0.iloc[j] - 5,2)
                        if min <0: min = 0 
                        max = round(df_min0.iloc[j] + 5,2)
                        #st.write(max, min)
                        para = np.arange(min, max, (max-min)/50.0)  
                        col_list.append(column)
                        para2.append(para)
                        
                    if df_min0.iloc[j] >= 20 and df_min0.iloc[j] < 40:
                        min = round(df_min0.iloc[j] - 10,2)
                        if min <0: min = 0 
                        max = round(df_min0.iloc[j] + 10,2)
                        #st.write(max, min)
                        para = np.arange(min, max, (max-min)/50.0)  
                        col_list.append(column)
                        para2.append(para)
                        
                    if df_min0.iloc[j] >= 40:
                        min = round(df_min0.iloc[j] - 20,2)
                        if min <0: min = 0 
                        max = round(df_min0.iloc[j] + 20,2)
                        #st.write(max, min)
                        para = np.arange(min, max, (max-min)/50.0)  
                        col_list.append(column)
                        para2.append(para)
                                                  
                    para2 = pd.DataFrame(para2)
                    para2 = para2.T
                    #st.write(col_list)
                    para6 = para2
                    para6.columns = col_list
                    
                    #st.write(para6)
            
            
                        
                                          
                    #st.write(para61)
                    
                    New_x2 = pd.DataFrame(X.iloc[0,:])
                    New_x2 = New_x2.T
                    
            
                    para7 = []
                    random.seed(42)
                    for i in range(5000):
                        para5 = []
                        for col in New_x2.columns:
                            New_x2[col] = 0.0
                                               
                            for col1 in list(para6.columns):
                                if col1 == col:
                                    New_x2[col] = random.sample(list(para6[col1]),1)
                                        
                                                              
                            para5.append(float(New_x2[col].values))
                          
                        para7.append(para5)
                               
            
                    para7 = pd.DataFrame(para7, columns=X.columns) 
                    para7 = para7.replace(np.nan, 0)
                    
                                       
            
            
                    
                    #para7 = para7.drop_duplicates()
                    #para7 = para7.reset_index(drop=True)
                    
            
                    datafile2 = para7.values
                    
                    #rescaleddatafile2 = scaler.transform(datafile2)
                       
                    predictions3 = model.predict(datafile2)
               
                    predictions3 = pd.DataFrame(predictions3, columns=['Pred_L','Pred_a','Pred_b'])
                       
                    para7 = pd.concat([para7,predictions3], axis=1)
                           
                    para7 = para7.reset_index(drop = True)
                    
                    para7['Delta_E'] = 0.0
                                      
                    for i in range(para7.shape[0]):
            
                        para7['Delta_E'][i] = ((Target_v['Target_L'] - predictions3['Pred_L'][i])**2+(Target_v['Target_a'] - predictions3['Pred_a'][i])**2+(Target_v['Target_b'] - predictions3['Pred_b'][i])**2)**0.5 
                    
                    para7.sort_values(by='Delta_E', ascending=True, inplace =True)
                    para7 = para7.head(1)
                    para77 = para4[:1]
                    
                    
                    para7 = pd.concat([para7,para77], axis=0)
                    para7.sort_values(by='Delta_E', ascending=True, inplace =True)
                    
                    para7 = para7.head(1)
            
                    #st.write(para7)
                    
                    
                    
                    
                    para7 = para7.reset_index(drop=True)
                    
                    
                    
                    """st.write('**2??? ?????? ????????? ??????:**')
                    
                    df_min0 = para4.head(3)
                    df_min1 = para7.head(3)
                    
                    df_min11 = pd.concat([df_min0,df_min1], axis=0)
                    
                    df_min11.sort_values(by='Delta_E', ascending=True, inplace =True)
                    
                    
                    
                    st.write(para7)"""
                    
                    
                    
                    
                    df_min3 = para7.iloc[0]
                    
                    
                    
                    
            
                    para23 =[]
                    para63 = pd.DataFrame()
            
                    col_list = []
                    for j in range(df_min3.shape[0]-4):
                            
                        column = df_min3.index[j]
                        
                        
                        if df_min3.iloc[j] > 0 and df_min3.iloc[j] < 20:
                            min = round(df_min3.iloc[j] - 2,3)
                            if min <0: min = 0 
                            max = round(df_min3.iloc[j] + 2,3)
                            #st.write(max, min)
                            para = np.arange(min, max, (max-min)/50.0)  
                            col_list.append(column)
                            para23.append(para)
                            
                        if df_min3.iloc[j] >= 20:
                            min = round(df_min3.iloc[j] - 4,3)
                            if min <0: min = 0 
                            max = round(df_min3.iloc[j] + 4,3)
                            #st.write(max, min)
                            para = np.arange(min, max, (max-min)/50.0)  
                            col_list.append(column)
                            para23.append(para)
                                                  
                    para23 = pd.DataFrame(para23)
                    para23 = para23.T
                    #st.write(col_list)
                    para63 = para23
                    para63.columns = col_list
                    
                    
                    

                                     
                    
                    
              
            
                        
                    #st.write(para61)
                    
                    New_x2 = pd.DataFrame(X.iloc[0,:])
                    New_x2 = New_x2.T
                    
                    para8 = []
                    random.seed(42)
                    for i in range(5000):
                        para5 = []
                        for col in New_x2.columns:
                            New_x2[col] = 0.0
                                               
                            for col1 in list(para63.columns):
                                if col1 == col:
                                    New_x2[col] = random.sample(list(para63[col1]),1)
                                        
                                                              
                            para5.append(float(New_x2[col].values))
                          
                        para8.append(para5)
                               
            
                    para8 = pd.DataFrame(para8, columns=X.columns) 
                    
                      
                   
            
            
                    datafile3 = para8.values
            
                    #rescaleddatafile3 = scaler.transform(datafile3)
                       
                    predictions4 = model.predict(datafile3)
               
                    predictions4 = pd.DataFrame(predictions4, columns=['Pred_L','Pred_a','Pred_b'])
                       
                    para8 = pd.concat([para8,predictions4], axis=1)
                       
                    para8 = para8.reset_index(drop = True)
                    
                    para8['Delta_E'] = 0.0
                                      
                    for i in range(para8.shape[0]):
            
                        para8['Delta_E'][i] = ((Target_v['Target_L'] - predictions4['Pred_L'][i])**2+(Target_v['Target_a'] - predictions4['Pred_a'][i])**2+(Target_v['Target_b'] - predictions4['Pred_b'][i])**2)**0.5 
                    
                    para8.sort_values(by='Delta_E', ascending=True, inplace =True)
                    
                    para8 = para8.head(1)
                    para88 = para7[:1]
                    
                    
                    para8 = pd.concat([para8,para88], axis=0)
                    para8.sort_values(by='Delta_E', ascending=True, inplace =True)
                    
                    para8 = para8.head(1)
            
                    #st.write(para8)
                    
                    
            
                    
                    st.markdown("<h6 style='text-align: left; color: darkblue;'> ??? ?????? ?????? ?????? ?????? </h6>", unsafe_allow_html=True)
                    
                    
                    para8 = para8.reset_index(drop=True)
                    
                    
                    r, g, b = color_change(para8['Pred_L'][0] ,para8['Pred_a'][0] , para8['Pred_b'][0])
                    
                    img = Image.new("RGB", (250, 50), color=(r,g,b))
                    st.image(img, caption='Prediction')
                    
                    
                    
                    para8['Delta_L'] = 0.0
                    para8['Delta_a'] = 0.0
                    para8['Delta_b'] = 0.0
                    
                    for i in range(para8.shape[0]):
                        para8['Delta_L'][i] = (Target_v['Target_L'] - para8['Pred_L'][i]) 
                        para8['Delta_a'][i] = (Target_v['Target_a'] - para8['Pred_a'][i])
                        para8['Delta_b'][i] = (Target_v['Target_b'] - para8['Pred_b'][i])
                        
                    #st.write(para8)
                        
                    para8 = para8.drop(['Pred_L','Pred_a','Pred_b'], axis=1)
                    
                    
                    #st.write(para8)
                    
                    para8_new = para8.replace(0,np.nan)
                    para8_new = pd.DataFrame(para8_new)
                    para8_new = para8_new.dropna(how='all',axis=1)
                    para8_new = para8_new.replace(np.nan,0)
                    
                    
                    para8_new2 = para8_new.drop(['Delta_E'], axis=1)
                    para8_new2['Delta_E'] = para8_new['Delta_E']
                    para8_new = para8_new2
                    

                    
                    
                    st.write(para8_new)
                    
                    

                    
                    
                    
                if new_df.shape[0]==2:
                            
                    st.write('')
                    st.write('')
                    
                    df_min0 = df_min.iloc[0]
                    df_min1 = df_min.iloc[1]
                    
                    #df_min0 = pd.DataFrame(df_min0)
                    #test = df_min0[df_min0 !=0]
                    #st.write(test)
                    #st.write(df_min0)
                    #st.write(df_min0.shape)
            
                    
                    
                    para2 =[]
                    para6 = pd.DataFrame()
            
                    col_list = []
                    for j in range(df_min0.shape[0]-4):
                            
                        column = df_min0.index[j]
                        
             
                    if df_min0.iloc[j] > 0 and df_min0.iloc[j] < 20:
                        min = round(df_min0.iloc[j] - 5,2)
                        if min <0: min = 0 
                        max = round(df_min0.iloc[j] + 5,2)
                        #st.write(max, min)
                        para = np.arange(min, max, (max-min)/50.0)  
                        col_list.append(column)
                        para2.append(para)
                        
                    if df_min0.iloc[j] >= 20 and df_min0.iloc[j] < 40:
                        min = round(df_min0.iloc[j] - 10,2)
                        if min <0: min = 0 
                        max = round(df_min0.iloc[j] + 10,2)
                        #st.write(max, min)
                        para = np.arange(min, max, (max-min)/50.0)  
                        col_list.append(column)
                        para2.append(para)
                        
                    if df_min0.iloc[j] >= 40:
                        min = round(df_min0.iloc[j] - 20,2)
                        if min <0: min = 0 
                        max = round(df_min0.iloc[j] + 20,2)
                        #st.write(max, min)
                        para = np.arange(min, max, (max-min)/50.0)  
                        col_list.append(column)
                        para2.append(para)
                                                  
                    para2 = pd.DataFrame(para2)
                    para2 = para2.T
                    #st.write(col_list)
                    para6 = para2
                    para6.columns = col_list
                    
                    #st.write(para6)
            
            
                        
                    
                    para21 =[]
                    para61 = pd.DataFrame()
            
                    col_list1 = []
                    for j in range(df_min1.shape[0]-4):
                            
                        column = df_min1.index[j]
            
                            
                    if df_min0.iloc[j] > 0 and df_min0.iloc[j] < 20:
                        min = round(df_min0.iloc[j] - 5,2)
                        if min <0: min = 0 
                        max = round(df_min0.iloc[j] + 5,2)
                        #st.write(max, min)
                        para = np.arange(min, max, (max-min)/50.0)  
                        col_list1.append(column)
                        para21.append(para)

                    if df_min0.iloc[j] >= 20 and df_min0.iloc[j] < 40:
                        min = round(df_min0.iloc[j] - 10,2)
                        if min <0: min = 0 
                        max = round(df_min0.iloc[j] + 10,2)
                        #st.write(max, min)
                        para = np.arange(min, max, (max-min)/50.0)  
                        col_list1.append(column)
                        para21.append(para)

                        
                    if df_min0.iloc[j] >= 40:
                        min = round(df_min0.iloc[j] - 20,2)
                        if min <0: min = 0 
                        max = round(df_min0.iloc[j] + 20,2)
                        #st.write(max, min)
                        para = np.arange(min, max, (max-min)/50.0)  
                        col_list1.append(column)
                        para21.append(para)
                                
                    para21 = pd.DataFrame(para21)
                    para21 = para21.T
                    #st.write(col_list1)
                    para61 = para21
                    para61.columns = col_list1
            
                    
                        
                        
                    #st.write(para61)
                    
                    New_x2 = pd.DataFrame(X.iloc[0,:])
                    New_x2 = New_x2.T
                    
            
                    para7 = []
                    random.seed(42)
                    for i in range(5000):
                        para5 = []
                        for col in New_x2.columns:
                            New_x2[col] = 0.0
                                               
                            for col1 in list(para6.columns):
                                if col1 == col:
                                    New_x2[col] = random.sample(list(para6[col1]),1)
                                        
                                                              
                            para5.append(float(New_x2[col].values))
                          
                        para7.append(para5)
                               
            
                    para7 = pd.DataFrame(para7, columns=X.columns) 
                    para7 = para7.replace(np.nan, 0)
                    
                    
                    
                    para71 = []
                    random.seed(42)
                    for i in range(5000):
                        para5 = []
                        for col in New_x2.columns:
                            New_x2[col] = 0.0
                                               
                            for col1 in list(para61.columns):
                                if col1 == col:
                                    New_x2[col] = random.sample(list(para61[col1]),1)
                                        
                                                              
                            para5.append(float(New_x2[col].values))
                          
                        para71.append(para5)
                               
            
                    para71 = pd.DataFrame(para71, columns=X.columns) 
                    para71 = para71.replace(np.nan, 0)
                    
                     
            
            
                    
                    #para7 = para7.drop_duplicates()
                    #para7 = para7.reset_index(drop=True)
                    
            
                    datafile2 = para7.values
                    
                    #rescaleddatafile2 = scaler.transform(datafile2)
                       
                    predictions3 = model.predict(datafile2)
               
                    predictions3 = pd.DataFrame(predictions3, columns=['Pred_L','Pred_a','Pred_b'])
                       
                    para7 = pd.concat([para7,predictions3], axis=1)
                           
                    para7 = para7.reset_index(drop = True)
                    
                    para7['Delta_E'] = 0.0
                                      
                    for i in range(para7.shape[0]):
            
                        para7['Delta_E'][i] = ((Target_v['Target_L'] - predictions3['Pred_L'][i])**2+(Target_v['Target_a'] - predictions3['Pred_a'][i])**2+(Target_v['Target_b'] - predictions3['Pred_b'][i])**2)**0.5 
                    
                    para7.sort_values(by='Delta_E', ascending=True, inplace =True)
                    para7 = para7.head(1)
                    para77 = para4[:1]
                    
                    
                    para7 = pd.concat([para7,para77], axis=0)
                    para7.sort_values(by='Delta_E', ascending=True, inplace =True)
                    
                    para7 = para7.head(1)
            
                    #st.write(para7)
                    
                    
                    
                    
                    datafile2 = para71.values
                    
                    #rescaleddatafile2 = scaler.transform(datafile2)
                       
                    predictions3 = model.predict(datafile2)
               
                    predictions3 = pd.DataFrame(predictions3, columns=['Pred_L','Pred_a','Pred_b'])
                       
                    para71 = pd.concat([para71,predictions3], axis=1)
                           
                    para71 = para71.reset_index(drop = True)
                    
                    para71['Delta_E'] = 0.0
                                      
                    for i in range(para71.shape[0]):
            
                        para71['Delta_E'][i] = ((Target_v['Target_L'] - predictions3['Pred_L'][i])**2+(Target_v['Target_a'] - predictions3['Pred_a'][i])**2+(Target_v['Target_b'] - predictions3['Pred_b'][i])**2)**0.5 
                    
                    para71.sort_values(by='Delta_E', ascending=True, inplace =True)
                    
                    para71 = para71.head(1)
                    para717 = para4[1:2]
                    
                    
                    para71 = pd.concat([para71,para717], axis=0)
                    para71.sort_values(by='Delta_E', ascending=True, inplace =True)
                    
                    para71 = para71.head(1)
                    #st.write(para71)
                    
                    
                         
                    
                    
                    #st.markdown("<h6 style='text-align: left; color: darkblue;'> 2??? ?????? ?????? ?????? </h6>", unsafe_allow_html=True)
                    
                    para7 = pd.concat([para7,para71], axis=0)
                    
                    para7.sort_values(by='Delta_E', ascending=True, inplace =True)
                    
                    para7 = para7.reset_index(drop=True)
                    
                    
                    
                    """st.write('**2??? ?????? ????????? ??????:**')
                    
                    df_min0 = para4.head(3)
                    df_min1 = para7.head(3)
                    
                    df_min11 = pd.concat([df_min0,df_min1], axis=0)
                    
                    df_min11.sort_values(by='Delta_E', ascending=True, inplace =True)
                    
                    
                    
                    st.write(para7)"""
                    
                    
                    
                    
                    df_min3 = para7.iloc[0]
                    df_min4 = para7.iloc[1]
                    
                    
                    
            
                    para23 =[]
                    para63 = pd.DataFrame()
            
                    col_list = []
                    for j in range(df_min3.shape[0]-4):
                            
                        column = df_min3.index[j]
                        
                        if df_min3.iloc[j] > 0 and df_min3.iloc[j] < 20:
                            min = round(df_min3.iloc[j] - 2,3)
                            if min <0: min = 0 
                            max = round(df_min3.iloc[j] + 2,3)
                            #st.write(max, min)
                            para = np.arange(min, max, (max-min)/50.0)  
                            col_list.append(column)
                            para23.append(para)
                            
                        if df_min3.iloc[j] >= 20:
                            min = round(df_min3.iloc[j] - 4,3)
                            if min <0: min = 0 
                            max = round(df_min3.iloc[j] + 4,3)
                            #st.write(max, min)
                            para = np.arange(min, max, (max-min)/50.0)  
                            col_list.append(column)
                            para23.append(para)
                                                  
                    para23 = pd.DataFrame(para23)
                    para23 = para23.T
                    #st.write(col_list)
                    para63 = para23
                    para63.columns = col_list
                    
                    
                    
                    para24 =[]
                    para64 = pd.DataFrame()
            
                    col_list = []
                    for j in range(df_min4.shape[0]-4):
                            
                        column = df_min4.index[j]
                        
                        if df_min4.iloc[j] > 0 and df_min4.iloc[j] < 20:
                            min = round(df_min4.iloc[j] - 2,3)
                            if min <0: min = 0 
                            max = round(df_min4.iloc[j] + 2,3)
                            #st.write(max, min)
                            para = np.arange(min, max, (max-min)/50.0)  
                            col_list.append(column)
                            para24.append(para)
                            
                        if df_min4.iloc[j] >= 20:
                            min = round(df_min4.iloc[j] - 4,3)
                            if min <0: min = 0 
                            max = round(df_min4.iloc[j] + 4,3)
                            #st.write(max, min)
                            para = np.arange(min, max, (max-min)/50.0)  
                            col_list.append(column)
                            para24.append(para)
                                                  
                    para24 = pd.DataFrame(para24)
                    para24 = para24.T
                    #st.write(col_list)
                    para64 = para24
                    para64.columns = col_list
                    
                    
                    
              
            
                        
                    #st.write(para61)
                    
                    New_x2 = pd.DataFrame(X.iloc[0,:])
                    New_x2 = New_x2.T
                    
                    para8 = []
                    random.seed(42)
                    for i in range(5000):
                        para5 = []
                        for col in New_x2.columns:
                            New_x2[col] = 0.0
                                               
                            for col1 in list(para63.columns):
                                if col1 == col:
                                    New_x2[col] = random.sample(list(para63[col1]),1)
                                        
                                                              
                            para5.append(float(New_x2[col].values))
                          
                        para8.append(para5)
                               
            
                    para8 = pd.DataFrame(para8, columns=X.columns) 
                    
                    para8 = para8.replace(np.nan, 0)
                    
                    
                    para81 = []
                    random.seed(42)
                    for i in range(5000):
                        para5 = []
                        for col in New_x2.columns:
                            New_x2[col] = 0.0
                                               
                            for col1 in list(para64.columns):
                                if col1 == col:
                                    New_x2[col] = random.sample(list(para64[col1]),1)
                                        
                                                              
                            para5.append(float(New_x2[col].values))
                          
                        para81.append(para5)
                               
            
                    para81 = pd.DataFrame(para81, columns=X.columns) 
                    
                    para81 = para81.replace(np.nan, 0)
                   
            
            
                    datafile3 = para8.values
            
                    #rescaleddatafile3 = scaler.transform(datafile3)
                       
                    predictions4 = model.predict(datafile3)
               
                    predictions4 = pd.DataFrame(predictions4, columns=['Pred_L','Pred_a','Pred_b'])
                       
                    para8 = pd.concat([para8,predictions4], axis=1)
                       
                    para8 = para8.reset_index(drop = True)
                    
                    para8['Delta_E'] = 0.0
                                      
                    for i in range(para8.shape[0]):
            
                        para8['Delta_E'][i] = ((Target_v['Target_L'] - predictions4['Pred_L'][i])**2+(Target_v['Target_a'] - predictions4['Pred_a'][i])**2+(Target_v['Target_b'] - predictions4['Pred_b'][i])**2)**0.5 
                    
                    para8.sort_values(by='Delta_E', ascending=True, inplace =True)
                    
                    para8 = para8.head(1)
                    para88 = para7[:1]
                    
                    
                    para8 = pd.concat([para8,para88], axis=0)
                    para8.sort_values(by='Delta_E', ascending=True, inplace =True)
                    
                    para8 = para8.head(1)
            
                    #st.write(para8)
                    
                    
                    
                    datafile3 = para81.values
            
                    #rescaleddatafile3 = scaler.transform(datafile3)
                       
                    predictions4 = model.predict(datafile3)
               
                    predictions4 = pd.DataFrame(predictions4, columns=['Pred_L','Pred_a','Pred_b'])
                       
                    para81 = pd.concat([para81,predictions4], axis=1)
                       
                    para81 = para81.reset_index(drop = True)
                    
                    para81['Delta_E'] = 0.0
                                      
                    for i in range(para81.shape[0]):

            
                        para81['Delta_E'][i] = ((Target_v['Target_L'] - predictions4['Pred_L'][i])**2+(Target_v['Target_a'] - predictions4['Pred_a'][i])**2+(Target_v['Target_b'] - predictions4['Pred_b'][i])**2)**0.5 
                    
                    para81.sort_values(by='Delta_E', ascending=True, inplace =True)
                    
                    para81 = para81.head(1)
                    para818 = para7[1:2]
                    
                    
                    para81 = pd.concat([para81,para818], axis=0)
                    para81.sort_values(by='Delta_E', ascending=True, inplace =True)
                    
                    para81 = para81.head(1)
            
                    #st.write(para81)

            
                    
                    st.markdown("<h6 style='text-align: left; color: darkblue;'> ??? ?????? ?????? ?????? ?????? </h6>", unsafe_allow_html=True)
                    
                    para8 = pd.concat([para8,para81], axis=0)
                    
                    para8.sort_values(by='Delta_E', ascending=True, inplace =True)
                    
                    para8 = para8.reset_index(drop=True)
                    
                    r, g, b = color_change(para8['Pred_L'][0] ,para8['Pred_a'][0] , para8['Pred_b'][0])
                    
                    img = Image.new("RGB", (250, 50), color=(r,g,b))
                    st.image(img, caption='Prediction')
                    
                    
                    
                    para8['Delta_L'] = 0.0
                    para8['Delta_a'] = 0.0
                    para8['Delta_b'] = 0.0
                    
                    for i in range(para8.shape[0]):
                        para8['Delta_L'][i] = (Target_v['Target_L'] - para8['Pred_L'][i]) 
                        para8['Delta_a'][i] = (Target_v['Target_a'] - para8['Pred_a'][i])
                        para8['Delta_b'][i] = (Target_v['Target_b'] - para8['Pred_b'][i])
                        
                    #st.write(para8)
                        
                    para8 = para8.drop(['Pred_L','Pred_a','Pred_b'], axis=1)
                    
                    #st.write(para8)
                    
                    para8_new = para8.replace(0,np.nan)
                    para8_new = pd.DataFrame(para8_new)
                    para8_new = para8_new.dropna(how='all',axis=1)
                    para8_new = para8_new.replace(np.nan,0)

                    para8_new2 = para8_new.drop(['Delta_E'], axis=1)
                    para8_new2['Delta_E'] = para8_new['Delta_E']
                    para8_new = para8_new2
                    
                                    
                    st.write(para8_new)
                    
                    

                    
                    
                    
                if new_df.shape[0] > 2:
                    
                    df_min = new_df.head(3)
            
                    st.write('')
                    st.write('')
                    
                    df_min0 = df_min.iloc[0]
                    df_min1 = df_min.iloc[1]
                    df_min2 = df_min.iloc[2]
                    
                    #df_min0 = pd.DataFrame(df_min0)
                    #test = df_min0[df_min0 !=0]
                    #st.write(test)
                    #st.write(df_min0)
                    #st.write(df_min0.shape)
            
                    
                    
                    para2 =[]
                    para6 = pd.DataFrame()
            
                    col_list = []
                    for j in range(df_min0.shape[0]-4):
                            
                        column = df_min0.index[j]
                        
                    if df_min0.iloc[j] > 0 and df_min0.iloc[j] < 20:
                        min = round(df_min0.iloc[j] - 5,2)
                        if min <0: min = 0 
                        max = round(df_min0.iloc[j] + 5,2)
                        #st.write(max, min)
                        para = np.arange(min, max, (max-min)/50.0)  
                        col_list.append(column)
                        para2.append(para)

                    if df_min0.iloc[j] >= 20 and df_min0.iloc[j] < 40:
                        min = round(df_min0.iloc[j] - 10,2)
                        if min <0: min = 0 
                        max = round(df_min0.iloc[j] + 10,2)
                        #st.write(max, min)
                        para = np.arange(min, max, (max-min)/50.0)  
                        col_list.append(column)
                        para2.append(para)

                    if df_min0.iloc[j] >= 40:
                        min = round(df_min0.iloc[j] - 20,2)
                        if min <0: min = 0 
                        max = round(df_min0.iloc[j] + 20,2)
                        #st.write(max, min)
                        para = np.arange(min, max, (max-min)/50.0)  
                        col_list.append(column)
                        para2.append(para)
                                                  
                    para2 = pd.DataFrame(para2)
                    para2 = para2.T
                    #st.write(col_list)
                    para6 = para2
                    para6.columns = col_list
                    
                    #st.write(para6)
            
            
                        
                    
                    para21 =[]
                    para61 = pd.DataFrame()
            
                    col_list1 = []
                    for j in range(df_min1.shape[0]-4):
                            
                        column = df_min1.index[j]
            
                    if df_min0.iloc[j] > 0 and df_min0.iloc[j] < 20:
                        min = round(df_min0.iloc[j] - 5,2)
                        if min <0: min = 0 
                        max = round(df_min0.iloc[j] + 5,2)
                        #st.write(max, min)
                        para = np.arange(min, max, (max-min)/50.0)  
                        col_list1.append(column)
                        para21.append(para)

                    if df_min0.iloc[j] >= 20 and df_min0.iloc[j] < 40:
                        min = round(df_min0.iloc[j] - 10,2)
                        if min <0: min = 0 
                        max = round(df_min0.iloc[j] + 10,2)
                        #st.write(max, min)
                        para = np.arange(min, max, (max-min)/50.0)  
                        col_list1.append(column)
                        para21.append(para)                        


                    if df_min0.iloc[j] >= 40:
                        min = round(df_min0.iloc[j] - 20,2)
                        if min <0: min = 0 
                        max = round(df_min0.iloc[j] + 20,2)
                        #st.write(max, min)
                        para = np.arange(min, max, (max-min)/50.0)  
                        col_list1.append(column)
                        para21.append(para)
                                
                    para21 = pd.DataFrame(para21)
                    para21 = para21.T
                    #st.write(col_list1)
                    para61 = para21
                    para61.columns = col_list1
            
                    
                    
                    para22 =[]
                    para62 = pd.DataFrame()
                    col_list2 = []
                    for j in range(df_min2.shape[0]-4):
                            
                        column = df_min2.index[j]
                        
                    if df_min0.iloc[j] > 0 and df_min0.iloc[j] < 20:
                        min = round(df_min0.iloc[j] - 5,2)
                        if min <0: min = 0 
                        max = round(df_min0.iloc[j] + 5,2)
                        #st.write(max, min)
                        para = np.arange(min, max, (max-min)/50.0)  
                        col_list2.append(column)
                        para22.append(para)

                        
                    if df_min0.iloc[j] >= 20 and df_min0.iloc[j] < 40:
                        min = round(df_min0.iloc[j] - 10,2)
                        if min <0: min = 0 
                        max = round(df_min0.iloc[j] + 10,2)
                        #st.write(max, min)
                        para = np.arange(min, max, (max-min)/50.0)  
                        col_list2.append(column)
                        para22.append(para)
                        
                        
                    if df_min0.iloc[j] >= 40:
                        min = round(df_min0.iloc[j] - 20,2)
                        if min <0: min = 0 
                        max = round(df_min0.iloc[j] + 20,2)
                        #st.write(max, min)
                        para = np.arange(min, max, (max-min)/50.0)  
                        col_list2.append(column)
                        para22.append(para)
                                
                    para22 = pd.DataFrame(para22)
                    para22 = para22.T
                    #st.write(col_list1)
                    para62 = para22
                    para62.columns = col_list2
            
                    
                    
                        
                        
                    #st.write(para61)
                    
                    New_x2 = pd.DataFrame(X.iloc[0,:])
                    New_x2 = New_x2.T
                    
            
                    para7 = []
                    random.seed(42)
                    for i in range(5000):
                        para5 = []
                        for col in New_x2.columns:
                            New_x2[col] = 0.0
                                               
                            for col1 in list(para6.columns):
                                if col1 == col:
                                    New_x2[col] = random.sample(list(para6[col1]),1)
                                        
                                                              
                            para5.append(float(New_x2[col].values))
                          
                        para7.append(para5)
                               
            
                    para7 = pd.DataFrame(para7, columns=X.columns) 
                    para7 = para7.replace(np.nan, 0)
                    
                    
                    
                    para71 = []
                    random.seed(42)
                    for i in range(5000):
                        para5 = []
                        for col in New_x2.columns:
                            New_x2[col] = 0.0
                                               
                            for col1 in list(para61.columns):
                                if col1 == col:
                                    New_x2[col] = random.sample(list(para61[col1]),1)
                                        
                                                              
                            para5.append(float(New_x2[col].values))
                          
                        para71.append(para5)
                               
            
                    para71 = pd.DataFrame(para71, columns=X.columns) 
                    para71 = para71.replace(np.nan, 0)
                    
                    
                    para72 = []
                    random.seed(42)
                    for i in range(500):
                        para5 = []
                        for col in New_x2.columns:
                            New_x2[col] = 0.0
                                               
                            for col1 in list(para62.columns):
                                if col1 == col:
                                    New_x2[col] = random.sample(list(para62[col1]),1)
                                        
                                                              
                            para5.append(float(New_x2[col].values))
                          
                        para72.append(para5)
                               
                    para72 = pd.DataFrame(para72, columns=X.columns)
                    para72 = para72.replace(np.nan, 0)
                    
            
            
                    
                    #para7 = para7.drop_duplicates()
                    #para7 = para7.reset_index(drop=True)
                    
            
                    datafile2 = para7.values
                    
                    #rescaleddatafile2 = scaler.transform(datafile2)
                       
                    predictions3 = model.predict(datafile2)
               
                    predictions3 = pd.DataFrame(predictions3, columns=['Pred_L','Pred_a','Pred_b'])
                       
                    para7 = pd.concat([para7,predictions3], axis=1)
                           
                    para7 = para7.reset_index(drop = True)
                    
                    para7['Delta_E'] = 0.0
                                      
                    for i in range(para7.shape[0]):
            
                        para7['Delta_E'][i] = ((Target_v['Target_L'] - predictions3['Pred_L'][i])**2+(Target_v['Target_a'] - predictions3['Pred_a'][i])**2+(Target_v['Target_b'] - predictions3['Pred_b'][i])**2)**0.5 
                    
                    para7.sort_values(by='Delta_E', ascending=True, inplace =True)
                    para7 = para7.head(1)
                    para77 = para4[:1]
                    
                    
                    para7 = pd.concat([para7,para77], axis=0)
                    para7.sort_values(by='Delta_E', ascending=True, inplace =True)
                    
                    para7 = para7.head(1)
            
                    #st.write(para7)
                    
                    
                    
                    
                    datafile2 = para71.values
                    
                    #rescaleddatafile2 = scaler.transform(datafile2)
                       
                    predictions3 = model.predict(datafile2)
               
                    predictions3 = pd.DataFrame(predictions3, columns=['Pred_L','Pred_a','Pred_b'])
                       
                    para71 = pd.concat([para71,predictions3], axis=1)
                           
                    para71 = para71.reset_index(drop = True)
                    
                    para71['Delta_E'] = 0.0
                                      
                    for i in range(para71.shape[0]):
            
                        para71['Delta_E'][i] = ((Target_v['Target_L'] - predictions3['Pred_L'][i])**2+(Target_v['Target_a'] - predictions3['Pred_a'][i])**2+(Target_v['Target_b'] - predictions3['Pred_b'][i])**2)**0.5 
                    
                    para71.sort_values(by='Delta_E', ascending=True, inplace =True)
                    
                    para71 = para71.head(1)
                    para717 = para4[1:2]
                    
                    
                    para71 = pd.concat([para71,para717], axis=0)
                    para71.sort_values(by='Delta_E', ascending=True, inplace =True)
                    
                    para71 = para71.head(1)
                    #st.write(para71)
                    
                    
                    
                    
                    datafile2 = para72.values
                    
                    #rescaleddatafile2 = scaler.transform(datafile2)
                       
                    predictions3 = model.predict(datafile2)
               
                    predictions3 = pd.DataFrame(predictions3, columns=['Pred_L','Pred_a','Pred_b'])
                       
                    para72 = pd.concat([para72,predictions3], axis=1)
                           
                    para72 = para72.reset_index(drop = True)
                    
                    para72['Delta_E'] = 0.0
                                      
                    for i in range(para72.shape[0]):
            
                        para72['Delta_E'][i] = ((Target_v['Target_L'] - predictions3['Pred_L'][i])**2+(Target_v['Target_a'] - predictions3['Pred_a'][i])**2+(Target_v['Target_b'] - predictions3['Pred_b'][i])**2)**0.5 
                    
                    para72.sort_values(by='Delta_E', ascending=True, inplace =True)
                    
                    para72 = para72.head(1)
                    para727 = para4[2:3]
                    
                    para72 = pd.concat([para72,para727], axis=0)
                    para72.sort_values(by='Delta_E', ascending=True, inplace =True)
                    
                    para72 = para72.head(1)
                    #st.write(para72)
                    
                    
                    
                    #st.markdown("<h6 style='text-align: left; color: darkblue;'> 2??? ?????? ?????? ?????? </h6>", unsafe_allow_html=True)
                    
                    para7 = pd.concat([para7,para71,para72], axis=0)
                    
                    para7.sort_values(by='Delta_E', ascending=True, inplace =True)
                    
                    para7 = para7.reset_index(drop=True)
                    
                    #st.write(para7)
                    
                    
                    """st.write('**2??? ?????? ????????? ??????:**')
                    
                    df_min0 = para4.head(3)
                    df_min1 = para7.head(3)
                    
                    df_min11 = pd.concat([df_min0,df_min1], axis=0)
                    
                    df_min11.sort_values(by='Delta_E', ascending=True, inplace =True)
                    
                    
                    
                    st.write(para7)"""
                    
                    
                    
                    
                    df_min3 = para7.iloc[0]
                    df_min4 = para7.iloc[1]
                    df_min5 = para7.iloc[2]
                    
                    
            
                    para23 =[]
                    para63 = pd.DataFrame()
            
                    col_list = []
                    for j in range(df_min3.shape[0]-4):
                            
                        column = df_min3.index[j]
                        
                        if df_min3.iloc[j] > 0 and df_min3.iloc[j] < 20:
                            min = round(df_min3.iloc[j] - 2,3)
                            if min <0: min = 0 
                            max = round(df_min3.iloc[j] + 2,3)
                            #st.write(max, min)
                            para = np.arange(min, max, (max-min)/50.0)  
                            col_list.append(column)
                            para23.append(para)
                            
                        if df_min3.iloc[j] >= 20:
                            min = round(df_min3.iloc[j] - 4,3)
                            if min <0: min = 0 
                            max = round(df_min3.iloc[j] + 4,3)
                            #st.write(max, min)
                            para = np.arange(min, max, (max-min)/50.0)  
                            col_list.append(column)
                            para23.append(para)
                                                  
                    para23 = pd.DataFrame(para23)
                    para23 = para23.T
                    #st.write(col_list)
                    para63 = para23
                    para63.columns = col_list
                    
                    
                    
                    para24 =[]
                    para64 = pd.DataFrame()
            
                    col_list = []
                    for j in range(df_min4.shape[0]-4):
                            
                        column = df_min4.index[j]
                        
                        if df_min4.iloc[j] > 0 and df_min4.iloc[j] < 20:
                            min = round(df_min4.iloc[j] - 2,3)
                            if min <0: min = 0 
                            max = round(df_min4.iloc[j] + 2,3)
                            #st.write(max, min)
                            para = np.arange(min, max, (max-min)/50.0)  
                            col_list.append(column)
                            para24.append(para)
                            
                        if df_min4.iloc[j] >= 20:
                            min = round(df_min4.iloc[j] - 4,3)
                            if min <0: min = 0 
                            max = round(df_min4.iloc[j] + 4,3)
                            #st.write(max, min)
                            para = np.arange(min, max, (max-min)/50.0)  
                            col_list.append(column)
                            para24.append(para)
                                                  
                    para24 = pd.DataFrame(para24)
                    para24 = para24.T
                    #st.write(col_list)
                    para64 = para24
                    para64.columns = col_list
                    
                    
                    
                    para25 =[]
                    para65 = pd.DataFrame()
            
                    col_list = []
                    for j in range(df_min5.shape[0]-4):
                            
                        column = df_min5.index[j]
                        
                        if df_min5.iloc[j] > 0 and df_min5.iloc[j] < 20:
                            min = round(df_min5.iloc[j] - 2,3)
                            if min <0: min = 0 
                            max = round(df_min5.iloc[j] + 2,3)
                            #st.write(max, min)
                            para = np.arange(min, max, (max-min)/50.0)  
                            col_list.append(column)
                            para25.append(para)
                            
                        if df_min5.iloc[j] >= 20:
                            min = round(df_min5.iloc[j] - 4,3)
                            if min <0: min = 0 
                            max = round(df_min5.iloc[j] + 4,3)
                            #st.write(max, min)
                            para = np.arange(min, max, (max-min)/50.0)  
                            col_list.append(column)
                            para25.append(para)
                                                  
                    para25 = pd.DataFrame(para25)
                    para25 = para25.T
                    #st.write(col_list)
                    para65 = para25
                    para65.columns = col_list
                    
             
            
                        
                    #st.write(para61)
                    
                    New_x2 = pd.DataFrame(X.iloc[0,:])
                    New_x2 = New_x2.T
                    
                    para8 = []
                    random.seed(42)
                    for i in range(5000):
                        para5 = []
                        for col in New_x2.columns:
                            New_x2[col] = 0.0
                                               
                            for col1 in list(para63.columns):
                                if col1 == col:
                                    New_x2[col] = random.sample(list(para63[col1]),1)
                                        
                                                              
                            para5.append(float(New_x2[col].values))
                          
                        para8.append(para5)
                               
            
                    para8 = pd.DataFrame(para8, columns=X.columns) 
                    para8 = para8.replace(np.nan, 0)
                    
                    
                    para81 = []
                    random.seed(42)
                    for i in range(5000):
                        para5 = []
                        for col in New_x2.columns:
                            New_x2[col] = 0.0
                                               
                            for col1 in list(para64.columns):
                                if col1 == col:
                                    New_x2[col] = random.sample(list(para64[col1]),1)
                                        
                                                              
                            para5.append(float(New_x2[col].values))
                          
                        para81.append(para5)
                               
            
                    para81 = pd.DataFrame(para81, columns=X.columns) 
                    para81 = para81.replace(np.nan, 0)
                    
                    
                    para82 = []
                    random.seed(42)
                    for i in range(500):
                        para5 = []
                        for col in New_x2.columns:
                            New_x2[col] = 0.0
                                               
                            for col1 in list(para65.columns):
                                if col1 == col:
                                    New_x2[col] = random.sample(list(para65[col1]),1)
                                        
                                                              
                            para5.append(float(New_x2[col].values))
                          
                        para82.append(para5)
                               
            
                    para82 = pd.DataFrame(para82, columns=X.columns)
                    para82 = para82.replace(np.nan, 0)
            
            
                    datafile3 = para8.values
            
                    #rescaleddatafile3 = scaler.transform(datafile3)
                       
                    predictions4 = model.predict(datafile3)
               
                    predictions4 = pd.DataFrame(predictions4, columns=['Pred_L','Pred_a','Pred_b'])
                       
                    para8 = pd.concat([para8,predictions4], axis=1)
                       
                    para8 = para8.reset_index(drop = True)
                    
                    para8['Delta_E'] = 0.0
                                      
                    for i in range(para8.shape[0]):
            
                        para8['Delta_E'][i] = ((Target_v['Target_L'] - predictions4['Pred_L'][i])**2+(Target_v['Target_a'] - predictions4['Pred_a'][i])**2+(Target_v['Target_b'] - predictions4['Pred_b'][i])**2)**0.5 
                    
                    para8.sort_values(by='Delta_E', ascending=True, inplace =True)
                    
                    para8 = para8.head(1)
                    para88 = para7[:1]
                    
                    
                    para8 = pd.concat([para8,para88], axis=0)
                    para8.sort_values(by='Delta_E', ascending=True, inplace =True)
                    
                    para8 = para8.head(1)
            
                    #st.write(para8)
                    
                    
                    
                    datafile3 = para81.values
            
                    #rescaleddatafile3 = scaler.transform(datafile3)
                       
                    predictions4 = model.predict(datafile3)
               
                    predictions4 = pd.DataFrame(predictions4, columns=['Pred_L','Pred_a','Pred_b'])
                       
                    para81 = pd.concat([para81,predictions4], axis=1)
                       
                    para81 = para81.reset_index(drop = True)
                    
                    para81['Delta_E'] = 0.0
                                      
                    for i in range(para81.shape[0]):
            
                        para81['Delta_E'][i] = ((Target_v['Target_L'] - predictions4['Pred_L'][i])**2+(Target_v['Target_a'] - predictions4['Pred_a'][i])**2+(Target_v['Target_b'] - predictions4['Pred_b'][i])**2)**0.5 
                    
                    para81.sort_values(by='Delta_E', ascending=True, inplace =True)
                    
                    para81 = para81.head(1)
                    para818 = para7[1:2]
                    
                    
                    para81 = pd.concat([para81,para818], axis=0)
                    para81.sort_values(by='Delta_E', ascending=True, inplace =True)
                    
                    para81 = para81.head(1)
            
                    #st.write(para81)
                    
                    
                    
                    datafile3 = para82.values
            
                    #rescaleddatafile3 = scaler.transform(datafile3)
                       
                    predictions4 = model.predict(datafile3)
               
                    predictions4 = pd.DataFrame(predictions4, columns=['Pred_L','Pred_a','Pred_b'])
                       
                    para82 = pd.concat([para82,predictions4], axis=1)
                       
                    para82 = para82.reset_index(drop = True)
                    
                    para82['Delta_E'] = 0.0
                                      
                    for i in range(para82.shape[0]):
            
                        para82['Delta_E'][i] = ((Target_v['Target_L'] - predictions4['Pred_L'][i])**2+(Target_v['Target_a'] - predictions4['Pred_a'][i])**2+(Target_v['Target_b'] - predictions4['Pred_b'][i])**2)**0.5 
                    
                    para82.sort_values(by='Delta_E', ascending=True, inplace =True)
                    
                    para82 = para82.head(1)
                    para828 = para7[2:3]
                    
                    
                    para82 = pd.concat([para82,para828], axis=0)
                    para82.sort_values(by='Delta_E', ascending=True, inplace =True)
                    
                    para82 = para82.head(1)
            
                    #st.write(para82)
                    
            
                    
                    st.markdown("<h6 style='text-align: left; color: darkblue;'> ??? ?????? ?????? ?????? ?????? </h6>", unsafe_allow_html=True)
                    
                    para8 = pd.concat([para8,para81,para82], axis=0)
                    
                    para8.sort_values(by='Delta_E', ascending=True, inplace =True)
                    
                    para8 = para8.reset_index(drop=True)
                    
                    
                    
                    r, g, b = color_change(para8['Pred_L'][0] ,para8['Pred_a'][0] , para8['Pred_b'][0])
                    img = Image.new("RGB", (250, 50), color=(r,g,b))
                    st.image(img, caption='Prediction')
                    
                    
                    
                    para8['Delta_L'] = 0.0
                    para8['Delta_a'] = 0.0
                    para8['Delta_b'] = 0.0
                    
                    for i in range(para8.shape[0]):
                        para8['Delta_L'][i] = (Target_v['Target_L'] - para8['Pred_L'][i]) 
                        para8['Delta_a'][i] = (Target_v['Target_a'] - para8['Pred_a'][i])
                        para8['Delta_b'][i] = (Target_v['Target_b'] - para8['Pred_b'][i])
                        
                    #st.write(para8)
                        
                    para8 = para8.drop(['Pred_L','Pred_a','Pred_b'], axis=1)

                    
                    para8_new = para8.replace(0,np.nan)
                    para8_new = pd.DataFrame(para8_new)
                    para8_new = para8_new.dropna(how='all',axis=1)
                    para8_new = para8_new.replace(np.nan,0)
                    
                    para8_new2 = para8_new.drop(['Delta_E'], axis=1)
                    para8_new2['Delta_E'] = para8_new['Delta_E']
                    para8_new = para8_new2                    
                        
                    
                    st.write(para8_new)
                    
 
                





    if ques == '???????????? ????????????':
        

        st.write('')
        st.write("<h5 style='text-align: left; color: black;'> 1.3 ?????? ???????????? - ???????????? ????????????</h5>", unsafe_allow_html=True)
        st.markdown("""<hr style="height:2px;border:none; color:rgb(60,90,180); background-color:rgb(60,90,180);"/> """, unsafe_allow_html=True)


        #col1,col2 = st.columns([1,1])
        
        col3,col4,col5,col6 = st.columns([1,1,1,3])
        
        columns = ['Target_L','Target_a','Target_b']

        Target_n = []
        Target_v = []
            
            
        with col3:
            value1 = st.number_input(columns[0], -1000.00, 1000.00, 0.0,format="%.3f")
            Target_n.append(columns[0])
            Target_v.append(value1)
        with col4:
            value2 = st.number_input(columns[1], -1000.00, 1000.00, 0.0,format="%.3f")
            Target_n.append(columns[1])
            Target_v.append(value2)
        with col5:
            value3 = st.number_input(columns[2], -1000.00, 1000.00, 0.0,format="%.3f")
            Target_n.append(columns[2])
            Target_v.append(value3)
                
            
        with col3:
            st.write('')
            select = ['YJ2442']
            selected1 = st.selectbox("?????? ?????? : ", select)
            
            


        

        
        color_list =[]   
        color_list = ['YY1137W', 'YY1120K', 'YY1116Y', 'YY1111R', 'YY1110R', 'YY1112B', 'YY1113B', 'YY1104G', 'YY1103G', 'YY1169R']
        color_list = pd.DataFrame(color_list,columns=['color'])
        
        DT = pd.read_csv('train_pcm.csv')
    
        count = 0
        
        para3 = pd.DataFrame()
        

        if Target_v[0] < 45.0 and Target_v[1] >= 0.0 and Target_v[2] >= 0.0:
            para3 = pd.read_csv('pcmdata001.csv')

            
        if Target_v[0] >= 45.0 and Target_v[0] < 65.0 and Target_v[1] >= 0.0 and Target_v[2] >= 0.0:
            para3 = pd.read_csv('pcmdata002.csv')

            
        if Target_v[0] >= 65.0 and Target_v[1] >= 0.0 and Target_v[2] >= 0.0:
            para3 = pd.read_csv('pcmdata003.csv')

            
        if Target_v[0] < 45.0 and Target_v[1] >= 0.0 and Target_v[2] < 0.0:
            para3 = pd.read_csv('pcmdata00_1.csv')


        if Target_v[0] >= 45.0 and Target_v[0] < 65.0 and Target_v[1] >= 0.0 and Target_v[2] < 0.0:
            para3 = pd.read_csv('pcmdata00_2.csv')


        if Target_v[0] >= 65.0 and Target_v[1] >= 0.0 and Target_v[2] < 0.0:
            para3 = pd.read_csv('pcmdata00_3.csv')
            
            

        if Target_v[0] < 45.0 and Target_v[1] < 0.0 and Target_v[2] >= 0.0:
            para3 = pd.read_csv('pcmdata0_01.csv')


        if Target_v[0] >= 45.0 and Target_v[0] < 65.0 and Target_v[1] < 0.0 and Target_v[2] >= 0.0:
            para3 = pd.read_csv('pcmdata0_02.csv')


        if Target_v[0] >= 65.0 and Target_v[1] < 0.0 and Target_v[2] >= 0.0:
            para3 = pd.read_csv('pcmdata0_03.csv')
   
            

        if Target_v[0] < 45.0 and Target_v[1] < 0.0 and Target_v[2] < 0.0:
            para3 = pd.read_csv('pcmdata0_0_1.csv')


        if Target_v[0] >= 45.0 and Target_v[0] < 65.0 and Target_v[1] < 0.0 and Target_v[2] < 0.0:
            para3 = pd.read_csv('pcmdata0_0_2.csv')


        if Target_v[0] >= 65.0 and Target_v[1] < 0.0 and Target_v[2] < 0.0:
            para3 = pd.read_csv('pcmdata0_0_3.csv')

        

        
        #color_list2 = para3[para3.columns != 0.0].columns
        
        #st.write(color_list2)
        
        col7,col8,col9 = st.columns([1,2,3])
        
        with col7:
            
            color_selected = []
            color_selected = st.multiselect("????????? ?????? : ", color_list)
    
        
        name = []
        test = [] 
        iter = 0
        
        with col8:
            for column in color_selected:

                max1 = round(float(para3[column].max()),3)
                if max1==0.0: max1 = 20.0

                min1 = round(float(para3[column].min()),3)
                   
                step = round((max1-min1)/20.0,3)
            
                value = st.slider(column, min1, max1, (min1,max1), step,key=11)
                     
                name.append(column)
                test.append(value)
                
                
                
        st.markdown("""<hr style="height:2px;border:none; color:rgb(60,90,180); background-color:rgb(60,90,180);"/> """, unsafe_allow_html=True)
        
        
        
    
        if st.button('Run Prediction',key = count):
            
            if Target_v[0] < 45.0 and Target_v[1] >= 0.0 and Target_v[2] >= 0.0:
                model = model1
                para3 = pd.read_csv('pcmdata001.csv')                

                
            if Target_v[0] >= 45.0 and Target_v[0] < 65.0 and Target_v[1] >= 0.0 and Target_v[2] >= 0.0:
                model = model2
                para3 = pd.read_csv('pcmdata002.csv')

                
            if Target_v[0] >= 65.0 and Target_v[1] >= 0.0 and Target_v[2] >= 0.0:
                model = model1
                para3 = pd.read_csv('pcmdata003.csv')

                
            if Target_v[0] < 45.0 and Target_v[1] >= 0.0 and Target_v[2] < 0.0:
                model = model4
                para3 = pd.read_csv('pcmdata00_1.csv')


            if Target_v[0] >= 45.0 and Target_v[0] < 65.0 and Target_v[1] >= 0.0 and Target_v[2] < 0.0:
                model = model1
                para3 = pd.read_csv('pcmdata00_2.csv')


            if Target_v[0] >= 65.0 and Target_v[1] >= 0.0 and Target_v[2] < 0.0:
                model = model2
                para3 = pd.read_csv('pcmdata00_3.csv')
                
                

            if Target_v[0] < 45.0 and Target_v[1] < 0.0 and Target_v[2] >= 0.0:
                model = model1
                para3 = pd.read_csv('pcmdata0_01.csv')


            if Target_v[0] >= 45.0 and Target_v[0] < 65.0 and Target_v[1] < 0.0 and Target_v[2] >= 0.0:
                model = model3
                para3 = pd.read_csv('pcmdata0_02.csv')


            if Target_v[0] >= 65.0 and Target_v[1] < 0.0 and Target_v[2] >= 0.0:
                model = model2
                para3 = pd.read_csv('pcmdata0_03.csv')
       

            if Target_v[0] < 45.0 and Target_v[1] < 0.0 and Target_v[2] < 0.0:
                model = model1
                para3 = pd.read_csv('pcmdata0_0_1.csv')


            if Target_v[0] >= 45.0 and Target_v[0] < 65.0 and Target_v[1] < 0.0 and Target_v[2] < 0.0:
                model = model2
                para3 = pd.read_csv('pcmdata0_0_2.csv')


            if Target_v[0] >= 65.0 and Target_v[1] < 0.0 and Target_v[2] < 0.0:
                model = model3
                para3 = pd.read_csv('pcmdata0_0_3.csv')
                
                
            
            
            
            with col3:

                st.write('')
                st.write('')
                r, g, b = color_change(Target_v[0] ,Target_v[1] , Target_v[2])
                
                img = Image.new("RGB", (250, 50), color=(r,g,b))
                st.image(img, caption='Target')
                
                
            
            with col6:
                
                
                st.markdown("<h6 style='text-align: left; color: darkblue;'> 1.3. ???????????? ???????????? </h6>", unsafe_allow_html=True)
                  
        
                #st.write(name)
                #st.write(test)
    
                para = []
                para2 = []
                para4 = []
            
                #st.write(test2)
                #import itertools
            
                for para in test:
                    if para[0] == para[1]:
                        para = itertools.repeat(para[0],100)
                    else:
                        para = np.arange(round(para[0],3), round(para[1]+((para[1]-para[0])/100.0),3), round((para[1]-para[0])/100.0,3))
                #st.write(para)
                    para2.append(para)
            
    
                #st.write(para2)
            
                para2 = pd.DataFrame(para2)
                para2 = para2.T
                para2 = para2.dropna().reset_index()
                para2.drop(['index'],axis=1,inplace=True)
            
                Iter2 = para2.shape[1]
    
                random.seed(42)
                for i in range(100):
                    para3 = []
                    para5 = []
                    for j in range(Iter2):
                        #st.write(i,j,list(para2[j]))
                        para3.append(random.sample(list(para2[j]),1))
                        para5.append(para3[j][0])
                    
                #para3 = pd.DataFrame(para3).values
                #para4.append(para3)
                    para4.append(para5)
                
                
            #para4 = pd.DataFrame(para4)
                para4 = pd.DataFrame(para4)
            
            
                para4.columns=list(name)
            
                st.markdown("<h6 style='text-align: left; color: darkblue;'> ****?????? ?????? ?????? ?????? ??? </h6>", unsafe_allow_html=True)
                #st.write(para4)
            
            
                New_x2 = pd.DataFrame(X.iloc[0,:])
                New_x2 = New_x2.T
                
        
                para7 = []
                
                random.seed(42)
                num = 1
                while num <=4000:
                    para5 = []
                    
                    for col in New_x2.columns:
                        New_x2[col] = 0.0
                                           
                        for col1 in list(para4.columns):
                            if col1 == col:
                                New_x2[col] = random.sample(list(para4[col1]),1)
                                                                                             
                        para5.append(float(New_x2[col].values))
                        
                        
                    if sum(para5) > 80.0 and sum(para5) < 90.0:
                        num = num+1
                        para7.append(para5)                        
                        
    
        
                para7 = pd.DataFrame(para7, columns=X.columns) 
                para7 = para7.replace(np.nan, 0)
                
                #st.write(para7)
                

    
    
                datafile2 = para7.values
                
                #rescaleddatafile2 = scaler.transform(datafile2)
                   
                predictions3 = model.predict(datafile2)
           
                predictions3 = pd.DataFrame(predictions3, columns=['Pred_L','Pred_a','Pred_b'])
                   
                para7 = pd.concat([para7,predictions3], axis=1)
                       
                para7 = para7.reset_index(drop = True)
                
                para7['Delta_E'] = 0.0
                
                
                Target_v = pd.DataFrame(Target_v)
                Target_v = Target_v.T
                Target_v.columns = Target_n
                
                para7 = para7.reset_index(drop=True)
                
                                  
                for i in range(para7.shape[0]):
        
                    para7['Delta_E'][i] = ((Target_v['Target_L'] - predictions3['Pred_L'][i])**2+(Target_v['Target_a'] - predictions3['Pred_a'][i])**2+(Target_v['Target_b'] - predictions3['Pred_b'][i])**2)**0.5 
                
                para7.sort_values(by='Delta_E', ascending=True, inplace =True)
                para7 = para7.head(3)
                
                #st.markdown("<h6 style='text-align: left; color: darkblue;'> ?????? ?????? ?????? ?????? </h6>", unsafe_allow_html=True)
                
                #st.write(para7)
    
                df_min3 = para7.iloc[0]
    
                
                
        
                para23 =[]
                para63 = pd.DataFrame()
        
                col_list = []
                for j in range(df_min3.shape[0]-4):
                        
                    column = df_min3.index[j]
                    
                    if df_min3.iloc[j] > 0 and df_min3.iloc[j] < 20:
                        min = round(df_min3.iloc[j] - 2,3)
                        if min <0: min = 0 
                        max = round(df_min3.iloc[j] + 2,3)
                        #st.write(max, min)
                        para = np.arange(min, max, (max-min)/50.0)  
                        col_list.append(column)
                        para23.append(para)
                        
                    if df_min3.iloc[j] >= 20:
                        min = round(df_min3.iloc[j] - 4,3)
                        if min <0: min = 0 
                        max = round(df_min3.iloc[j] + 4,3)
                        #st.write(max, min)
                        para = np.arange(min, max, (max-min)/50.0)  
                        col_list.append(column)
                        para23.append(para)
                                              
                para23 = pd.DataFrame(para23)
                para23 = para23.T
                #st.write(col_list)
                para63 = para23
                para63.columns = col_list
                
    
                    
                #st.write(para61)
                
                New_x2 = pd.DataFrame(X.iloc[0,:])
                New_x2 = New_x2.T
                
                para8 = []
                random.seed(42)
                for i in range(5000):
                    para5 = []
                    for col in New_x2.columns:
                        New_x2[col] = 0.0
                                           
                        for col1 in list(para63.columns):
                            if col1 == col:
                                New_x2[col] = random.sample(list(para63[col1]),1)
                                    
                                                          
                        para5.append(float(New_x2[col].values))
                      
                    para8.append(para5)
                           
        
                para8 = pd.DataFrame(para8, columns=X.columns) 
                para8 = para8.replace(np.nan, 0)
                
    
        
                datafile3 = para8.values
        
                #rescaleddatafile3 = scaler.transform(datafile3)
                   
                predictions4 = model.predict(datafile3)
           
                predictions4 = pd.DataFrame(predictions4, columns=['Pred_L','Pred_a','Pred_b'])
                   
                para8 = pd.concat([para8,predictions4], axis=1)
                   
                para8 = para8.reset_index(drop = True)
                
                para8['Delta_E'] = 0.0
                                  
                for i in range(para8.shape[0]):
        
                    para8['Delta_E'][i] = ((Target_v['Target_L'] - predictions4['Pred_L'][i])**2+(Target_v['Target_a'] - predictions4['Pred_a'][i])**2+(Target_v['Target_b'] - predictions4['Pred_b'][i])**2)**0.5 
                
                para8.sort_values(by='Delta_E', ascending=True, inplace =True)
                
                para8 = para8.head(1)
                para88 = para7[:1]
                
                
                para8 = pd.concat([para8,para88], axis=0)
                para8.sort_values(by='Delta_E', ascending=True, inplace =True)
                para8 = pd.DataFrame(para8)
                
                st.write('')
                
                para8 = para8.head(1)
                
 

                st.markdown("<h6 style='text-align: left; color: darkblue;'> ??? ?????? ?????? ????????? ?????? </h6>", unsafe_allow_html=True)
                
                
                r, g, b = color_change(float(para8['Pred_L'].values) ,float(para8['Pred_a'].values) , float(para8['Pred_b'].values))
                
                img = Image.new("RGB", (250, 50), color=(r,g,b))
                st.image(img, caption='Prediction')
                
                #st.write(Target_v)
                
                para8['Delta_L'] = 0.0 
                para8['Delta_a'] = 0.0
                para8['Delta_b'] = 0.0
                para8['Delta_L'] = Target_v['Target_L'].values - para8['Pred_L'].values
                para8['Delta_a'] = Target_v['Target_a'].values - para8['Pred_a'].values
                para8['Delta_b'] = Target_v['Target_b'].values - para8['Pred_b'].values
                
                #st.write(para8)
                    
                para8 = para8.drop(['Pred_L','Pred_a','Pred_b'], axis=1)
                               

                para8_new = para8.replace(0,np.nan)
                para8_new = pd.DataFrame(para8_new)
                para8_new = para8_new.dropna(how='all',axis=1)
                para8_new = para8_new.replace(np.nan,0)
                
                
                para8_new2 = para8_new.drop(['Delta_E'], axis=1)
                para8_new2['Delta_E'] = para8_new['Delta_E']
                para8_new = para8_new2                

                
                st.write(para8_new)
                









    st.sidebar.write("<h5 style='text-align: left; color: black;'> 2. ?????? ?????? ?????? ?????????</h5>", unsafe_allow_html=True)
        
  
    #st.markdown("<h6 style='text-align: left; color: darkblue;'> 2.2. ?????? ????????? ?????? ?????? </h6>", unsafe_allow_html=True)
    
    ques2 = st.sidebar.radio('????????? ??????',('???????????? ????????????_??????', '???????????? ????????????_??????'),key=2)
    
        
    
    if ques2 ==  '???????????? ????????????_??????' :
        

        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")


        st.write("<h5 style='text-align: left; color: black;'> 2.1. ?????? ?????? ?????? ??????_??????</h5>", unsafe_allow_html=True)
        
        st.markdown("""<hr style="height:2px;border:none; color:rgb(60,90,180); background-color:rgb(60,90,180);"/> """, unsafe_allow_html=True)
        
        
        col3,col4,col5,col6 = st.columns([1,1,1,3])
        
        
        
        columns1 = ['Target_L','Target_a','Target_b']
            
    
        Target_n1 = []
        Target_v1 = []
            
        #col1,col2,col3 = st.columns([1,1,1])
            
        with col3:

            value4 = st.number_input(columns1[0], -1000.00, 1000.00, 0.0,format="%.3f", key=1)
            Target_n1.append(columns1[0])
            Target_v1.append(value4)
        with col4:

            value5 = st.number_input(columns1[1], -1000.00, 1000.00, 0.0,format="%.3f", key=2)
            Target_n1.append(columns1[1])
            Target_v1.append(value5)
        with col5:

            value6 = st.number_input(columns1[2], -1000.00, 1000.00, 0.0,format="%.3f", key=3)
            Target_n1.append(columns1[2])
            Target_v1.append(value6)
            
        Target_x = pd.DataFrame([Target_v1],columns=list(Target_n1))
        
        
        columns2 = ['Delta_L','Delta_a','Delta_b']
        
        
        Target_n2 = []
        Target_v2 = []
        
        #col1,col2,col3 = st.columns([1,1,1])
            
        with col3:
            value7 = st.number_input(columns2[0], -1000.00, 1000.00, 0.0,format="%.3f", key=5)
            Target_n2.append(columns2[0])
            Target_v2.append(value7)
        with col4:
            value8 = st.number_input(columns2[1], -1000.00, 1000.00, 0.0,format="%.3f", key=6)
            Target_n2.append(columns2[1])
            Target_v2.append(value8)
        with col5:
            value9 = st.number_input(columns2[2], -1000.00, 1000.00, 0.0,format="%.3f", key=7)
            Target_n2.append(columns2[2])
            Target_v2.append(value9)
            
        Delta_x = pd.DataFrame([Target_v2],columns=list(Target_n2))
        
        Actual_x = pd.DataFrame()
        Actual_x['Target_L'] = Target_x['Target_L']  + Delta_x['Delta_L'] 
        Actual_x['Target_a'] = Target_x['Target_a']  + Delta_x['Delta_a'] 
        Actual_x['Target_b'] = Target_x['Target_b']  + Delta_x['Delta_b'] 
  
        New_x2 = pd.DataFrame(X.iloc[0,:])
                
        New_x2 = New_x2.T
    
        
    
        col7,col8,col9,col10 = st.columns([1,1,1,3])
        
        

        with col3:
            st.write('')
            select = ['YJ2442']
            selected3 = st.selectbox("?????? ?????? : ", select, key=10)
            

            
        
        with col4:
            st.write('')
            color_list = ['YY1137W', 'YY1120K', 'YY1116Y', 'YY1111R', 'YY1110R', 'YY1112B', 'YY1113B', 'YY1104G', 'YY1103G', 'YY1169R']
            colors = st.multiselect('????????? ??????',color_list, key=12)

        with col5:
            st.write('')
            Target_n = []
            Target_v = []
        
            for color1 in colors:
                value = st.number_input(color1,0.00, 500.00, 0.0,format="%.2f")
                Target_n.append(color1)
                Target_v.append(value)
        
            New_x = pd.DataFrame([Target_v],columns=list(Target_n))
            
        st.markdown("""<hr style="height:2px;border:none; color:rgb(60,90,180); background-color:rgb(60,90,180);"/> """, unsafe_allow_html=True)
        
        
        
            
        if st.button('Run Prediction', key=11): 
            
            if Target_v[0] < 45.0 and Target_v[1] >= 0.0 and Target_v[2] >= 0.0:
                model = model1
                para3 = pd.read_csv('pcmdata001.csv')                

                
            if Target_v[0] >= 45.0 and Target_v[0] < 65.0 and Target_v[1] >= 0.0 and Target_v[2] >= 0.0:
                model = model2
                para3 = pd.read_csv('pcmdata002.csv')

                
            if Target_v[0] >= 65.0 and Target_v[1] >= 0.0 and Target_v[2] >= 0.0:
                model = model1
                para3 = pd.read_csv('pcmdata003.csv')

                
            if Target_v[0] < 45.0 and Target_v[1] >= 0.0 and Target_v[2] < 0.0:
                model = model4
                para3 = pd.read_csv('pcmdata00_1.csv')


            if Target_v[0] >= 45.0 and Target_v[0] < 65.0 and Target_v[1] >= 0.0 and Target_v[2] < 0.0:
                model = model1
                para3 = pd.read_csv('pcmdata00_2.csv')


            if Target_v[0] >= 65.0 and Target_v[1] >= 0.0 and Target_v[2] < 0.0:
                model = model2
                para3 = pd.read_csv('pcmdata00_3.csv')
                
                

            if Target_v[0] < 45.0 and Target_v[1] < 0.0 and Target_v[2] >= 0.0:
                model = model1
                para3 = pd.read_csv('pcmdata0_01.csv')


            if Target_v[0] >= 45.0 and Target_v[0] < 65.0 and Target_v[1] < 0.0 and Target_v[2] >= 0.0:
                model = model3
                para3 = pd.read_csv('pcmdata0_02.csv')


            if Target_v[0] >= 65.0 and Target_v[1] < 0.0 and Target_v[2] >= 0.0:
                model = model2
                para3 = pd.read_csv('pcmdata0_03.csv')
       

            if Target_v[0] < 45.0 and Target_v[1] < 0.0 and Target_v[2] < 0.0:
                model = model1
                para3 = pd.read_csv('pcmdata0_0_1.csv')


            if Target_v[0] >= 45.0 and Target_v[0] < 65.0 and Target_v[1] < 0.0 and Target_v[2] < 0.0:
                model = model2
                para3 = pd.read_csv('pcmdata0_0_2.csv')


            if Target_v[0] >= 65.0 and Target_v[1] < 0.0 and Target_v[2] < 0.0:
                model = model3
                para3 = pd.read_csv('pcmdata0_0_3.csv')
                
            
            
            
            
            
            with col6:
                
                st.markdown("<h6 style='text-align: left; color: darkblue;'> 2.1. ???????????? ????????????_?????? </h6>", unsafe_allow_html=True)
                
                
                for col in New_x2.columns:
                    New_x2[col] = 0.0
                    for col2 in New_x.columns:
                        if col == col2:
                            New_x2[col] = New_x[col2].values
                                  
                
                
                
                New_x2.index = ['Old_case']        
                
                st.write("")
                
                #st.write(New_x2)
                
                
                st.markdown("<h6 style='text-align: left; color: darkblue;'> ??? ????????? ????????? ?????? </h6>", unsafe_allow_html=True)
                    
                #st.write(New_x2.style.format("{:.5}"))
                
                #scaler = StandardScaler().fit(X_train)
                
                New_x2_new = New_x2.replace(0,np.nan)
                New_x2_new = pd.DataFrame(New_x2_new)
                New_x2_new = New_x2_new.dropna(how='all',axis=1)
                New_x2_new = New_x2_new.replace(np.nan,0)
                
                st.write(New_x2_new)
                
                
            
                    

                
                predictions = model.predict(New_x2)
                            
                predictions = pd.DataFrame(predictions,columns = ['Pred_L','Pred_a','Pred_b'])
                
                #predictions.index = ['Results'] 
                                
                #st.write(predictions.style.format("{:.5}"))
        
                        
        
                NTarget_v = []
                NTarget_v.append(Target_x['Target_L'] + (predictions['Pred_L']-Actual_x['Target_L']))
                NTarget_v.append(Target_x['Target_a'] + (predictions['Pred_a']-Actual_x['Target_a']))
                NTarget_v.append(Target_x['Target_b'] + (predictions['Pred_b']-Actual_x['Target_b']))
                
                
                
                NTarget_v = pd.DataFrame(NTarget_v)
                NTarget_v = NTarget_v.T
                NTarget_v.columns = Target_n1
                
                Diff = Target_x - NTarget_v
                
                
                #st.write(Diff)
                #st.write(Actual_x)
                #st.write(predictions)
                #st.write(NTarget_v)
                
                #Diff = Target_v1 - NTarget_v
                
                #st.write(Diff)
                
                #df_min0 = New_x2.T
                
                df_min0 = New_x2.T
                
            
 
        
                para2 =[]
                para6 = pd.DataFrame()
                
                col_list = []
                for j in range(df_min0.shape[0]):
                        
                    column = df_min0.index[j]
                    
                    #st.write(column)
                    #st.write(df_min0.iloc[j])
                    
                    if df_min0.iloc[j].values > 0 and df_min0.iloc[j].values < 20:
                        
                        min1 = df_min0.iloc[j].values - 5
                        if min1 <0: min1 = 0 
                        
                        max1 = df_min0.iloc[j].values + 5
                        
                        para = np.arange(min1, max1, (max1-min1)/50.0)  
                        col_list.append(column)
                        para2.append(para)
                        
                        
                    if df_min0.iloc[j].values >= 20 and df_min0.iloc[j].values < 40:
                        
                        min1 = df_min0.iloc[j].values - 10
                        if min1 <0: min1 = 0 
                        
                        max1 = df_min0.iloc[j].values + 10
                        
                        para = np.arange(min1, max1, (max1-min1)/50.0)  
                        col_list.append(column)
                        para2.append(para)
                        
                        
                    if df_min0.iloc[j].values >= 40:
                        min1 = df_min0.iloc[j].values - 20
                        
                        if min1 <0: min1 = 0 
                        max1 = df_min0.iloc[j].values + 20
                        
                        para = np.arange(min1, max1, (max1-min1)/50.0)  
                        col_list.append(column)
                        para2.append(para)
                                              
                para2 = pd.DataFrame(para2)
                para2 = para2.T
                #st.write(col_list)
                para6 = para2
                para6.columns = col_list
                
                #st.write(para6)
                
                
                #st.write(df_min0.loc['SW1'])
                #st.write(df_min0.iloc[-1])
                
                
                # SW1 ??? ??????
                #if selected4 =='Base_C':
                #    para6['SW1'] = float(df_min0.loc['SW1'].values)
         
                
                
                New_x2 = pd.DataFrame(X.iloc[0,:])
                New_x2 = New_x2.T
                
                       
                
                para7 = []
                random.seed(42)
                for i in range(5000):
                    para5 = []
                    for col in New_x2.columns:
                        New_x2[col] = 0.0
                                           
                        for col1 in list(para6.columns):
                            if col1 == col:
                                New_x2[col] = random.sample(list(para6[col1]),1)
                                    
                                                          
                        para5.append(float(New_x2[col].values))
                      
                    para7.append(para5)
                
                
                para7 = pd.DataFrame(para7,columns=X.columns)
                
                para7 = para7.replace(np.nan,0.0)
                
                #st.write(para7)
                        
                datafile2 = para7.values
                   
                predictions3 = model.predict(datafile2)
           
                predictions3 = pd.DataFrame(predictions3, columns=['Pred_L','Pred_a','Pred_b'])
                   
                para71 = pd.concat([para7,predictions3], axis=1)
                       
                para71 = para71.reset_index(drop = True)
                
                para71['Delta_E'] = 0.0
                                  
                for i in range(para71.shape[0]):
        
                    para71['Delta_E'][i] = ((NTarget_v['Target_L'] - predictions3['Pred_L'][i])**2+(NTarget_v['Target_a'] - predictions3['Pred_a'][i])**2+(NTarget_v['Target_b'] - predictions3['Pred_b'][i])**2)**0.5 
                
                para71.sort_values(by='Delta_E', ascending=True, inplace =True)
                
                para71 = para71.head(3)
                
                #st.write(para71)
                
                
                
                
                
                
                df_min1 = para71.iloc[0]
                
                
                
                para22 =[]
                para62 = pd.DataFrame()
        
                col_list = []
                for j in range(df_min1.shape[0]-4):
                    
                    column = df_min1.index[j]
                    
                    #st.write(column)
                    
                    if df_min1.iloc[j] > 0:
                        
                        min1 = float(round(df_min1.iloc[j] - 2,3))
                        max1 = float(round(df_min1.iloc[j] + 2,3))
                        
                        if min1 < 0 : min1 = 0
                        
                        
                        para = np.arange(min1, max1, (max1-min1)/50.0)  
                        col_list.append(column)
                        para22.append(para)
                                              
          
                para22 = pd.DataFrame(para22)
                para22 = para22.T
                #st.write(col_list)
                para62 = para22
                para62.columns = col_list           
                
                
                #st.write(df_min1)
                
                
                # SW1 ??? ??????
                #if selected4 =='Base_C':
                #    para62['SW1'] = float(df_min1['SW1'])
                
        
        
        
                
                para72 = []
                random.seed(42)
                for i in range(5000):
                    para5 = []
                    for col in New_x2.columns:
                        New_x2[col] = 0.0
                                           
                        for col1 in list(para62.columns):
                            if col1 == col:
                                New_x2[col] = random.sample(list(para62[col1]),1)
                                                          
                        para5.append(float(New_x2[col].values))
                      
                    para72.append(para5)
                
                
                para72 = pd.DataFrame(para72,columns=X.columns)
                para72 = para72.replace(np.nan, 0)
                
                #st.write(para72)
                        
                datafile3 = para72.values
                   
                predictions4 = model.predict(datafile3)
           
                predictions4 = pd.DataFrame(predictions4, columns=['Pred_L','Pred_a','Pred_b'])
                   
                para73 = pd.concat([para72,predictions4], axis=1)
                       
                para73 = para73.reset_index(drop = True)
                
                para73['Delta_E'] = 0.0
                                  
                for i in range(para73.shape[0]):
        
                    para73['Delta_E'][i] = ((NTarget_v['Target_L'] - predictions4['Pred_L'][i])**2+(NTarget_v['Target_a'] - predictions4['Pred_a'][i])**2+(NTarget_v['Target_b'] - predictions4['Pred_b'][i])**2)**0.5 
                
                para73.sort_values(by='Delta_E', ascending=True, inplace =True)
                
                para73 = para73.head(1)        
                
                #st.write('1st=', para73)
                st.write('')
                st.write('')
                st.write('')
                st.write('')
                st.write('')
        
                st.markdown("<h6 style='text-align: left; color: darkblue;'> ??? ????????? ????????? ?????? </h6>", unsafe_allow_html=True)
        
        
                para73['Pred_L'] = para73['Pred_L'].values + Diff['Target_L'].values
                para73['Pred_a'] = para73['Pred_a'].values + Diff['Target_a'].values
                para73['Pred_b'] = para73['Pred_b'].values + Diff['Target_b'].values
                    
                
                #st.write(Target_v['Target_L'])
                #st.write(para73['Pred_L'].values)
                
                para73['Delta_L'] = 0.0 
                para73['Delta_a'] = 0.0
                para73['Delta_b'] = 0.0
                para73['Delta_L'] = Target_x['Target_L'].values - para73['Pred_L'].values
                para73['Delta_a'] = Target_x['Target_a'].values - para73['Pred_a'].values
                para73['Delta_b'] = Target_x['Target_b'].values - para73['Pred_b'].values
                
                #st.write(para73)
                
                para73 = para73.drop(['Pred_L','Pred_a','Pred_b'], axis=1)

                
                para73_new = para73.replace(0,np.nan)
                para73_new = pd.DataFrame(para73_new)
                para73_new = para73_new.dropna(how='all',axis=1)
                para73_new = para73_new.replace(np.nan,0)
                
                
                para73_new2 = para73_new.drop(['Delta_E'], axis=1)
                para73_new2['Delta_E'] = para73_new['Delta_E']
                para73_new = para73_new2
                
                
                st.write(para73_new)
                
                
                
                
    
                

    if ques2 ==  '???????????? ????????????_??????' :
        
        
        
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")

        st.write("<h5 style='text-align: left; color: black;'> 2.2. ?????? ?????? ?????? ??????_??????</h5>", unsafe_allow_html=True)
        st.markdown("""<hr style="height:2px;border:none; color:rgb(60,90,180); background-color:rgb(60,90,180);"/> """, unsafe_allow_html=True)
        
        
        
        col3,col4,col5,col6 = st.columns([1,1,1,3])
        
       
        

        columns1 = ['Target_L','Target_a','Target_b']
            
    
        Target_n1 = []
        Target_v1 = []
            
        #col1,col2,col3 = st.columns([1,1,1])
            
        with col3:

            value4 = st.number_input(columns1[0], -1000.00, 1000.00, 0.0,format="%.3f", key=1)
            Target_n1.append(columns1[0])
            Target_v1.append(value4)
        with col4:

            value5 = st.number_input(columns1[1], -1000.00, 1000.00, 0.0,format="%.3f", key=2)
            Target_n1.append(columns1[1])
            Target_v1.append(value5)
        with col5:

            value6 = st.number_input(columns1[2], -1000.00, 1000.00, 0.0,format="%.3f", key=3)
            Target_n1.append(columns1[2])
            Target_v1.append(value6)
            
        Target_x = pd.DataFrame([Target_v1],columns=list(Target_n1))
        
        
        columns2 = ['Delta_L','Delta_a','Delta_b']
        
        
        Target_n2 = []
        Target_v2 = []
        
        #col1,col2,col3 = st.columns([1,1,1])
            
        with col3:
            value7 = st.number_input(columns2[0], -1000.00, 1000.00, 0.0,format="%.3f", key=5)
            Target_n2.append(columns2[0])
            Target_v2.append(value7)
        with col4:
            value8 = st.number_input(columns2[1], -1000.00, 1000.00, 0.0,format="%.3f", key=6)
            Target_n2.append(columns2[1])
            Target_v2.append(value8)
        with col5:
            value9 = st.number_input(columns2[2], -1000.00, 1000.00, 0.0,format="%.3f", key=7)
            Target_n2.append(columns2[2])
            Target_v2.append(value9)
            
        Delta_x = pd.DataFrame([Target_v2],columns=list(Target_n2))
        
        Actual_x = pd.DataFrame()
        Actual_x['Target_L'] = Target_x['Target_L']  + Delta_x['Delta_L'] 
        Actual_x['Target_a'] = Target_x['Target_a']  + Delta_x['Delta_a'] 
        Actual_x['Target_b'] = Target_x['Target_b']  + Delta_x['Delta_b'] 
  
        New_x2 = pd.DataFrame(X.iloc[0,:])
                
        New_x2 = New_x2.T
    
        
    
        col7,col8,col9,col10 = st.columns([1,1,1,3])
        
        

        with col3:
            st.write('')
            select = ['YJ2442']
            selected3 = st.selectbox("?????? ?????? : ", select, key=10)
            
        
        with col4:
            st.write('')
            color_list = ['YY1137W', 'YY1120K', 'YY1116Y', 'YY1111R', 'YY1110R', 'YY1112B', 'YY1113B', 'YY1104G', 'YY1103G', 'YY1169R']
            colors = st.multiselect('????????? ??????',color_list, key=12)

        with col5:
            st.write('')
            Target_n = []
            Target_v = []
        
            for color1 in colors:
                value = st.number_input(color1,0.00, 500.00, 0.0,format="%.2f")
                Target_n.append(color1)
                Target_v.append(value)
        
            New_x = pd.DataFrame([Target_v],columns=list(Target_n))
            
        st.markdown("""<hr style="height:2px;border:none; color:rgb(60,90,180); background-color:rgb(60,90,180);"/> """, unsafe_allow_html=True)
        
        
   
            
        
            
        if st.button('Run Prediction', key=11): 
            
            
            if Target_v[0] < 45.0 and Target_v[1] >= 0.0 and Target_v[2] >= 0.0:
                model = model1
                para3 = pd.read_csv('pcmdata001.csv')                

                
            if Target_v[0] >= 45.0 and Target_v[0] < 65.0 and Target_v[1] >= 0.0 and Target_v[2] >= 0.0:
                model = model2
                para3 = pd.read_csv('pcmdata002.csv')

                
            if Target_v[0] >= 65.0 and Target_v[1] >= 0.0 and Target_v[2] >= 0.0:
                model = model1
                para3 = pd.read_csv('pcmdata003.csv')

                
            if Target_v[0] < 45.0 and Target_v[1] >= 0.0 and Target_v[2] < 0.0:
                model = model4
                para3 = pd.read_csv('pcmdata00_1.csv')


            if Target_v[0] >= 45.0 and Target_v[0] < 65.0 and Target_v[1] >= 0.0 and Target_v[2] < 0.0:
                model = model1
                para3 = pd.read_csv('pcmdata00_2.csv')


            if Target_v[0] >= 65.0 and Target_v[1] >= 0.0 and Target_v[2] < 0.0:
                model = model2
                para3 = pd.read_csv('pcmdata00_3.csv')
                
                

            if Target_v[0] < 45.0 and Target_v[1] < 0.0 and Target_v[2] >= 0.0:
                model = model1
                para3 = pd.read_csv('pcmdata0_01.csv')


            if Target_v[0] >= 45.0 and Target_v[0] < 65.0 and Target_v[1] < 0.0 and Target_v[2] >= 0.0:
                model = model3
                para3 = pd.read_csv('pcmdata0_02.csv')


            if Target_v[0] >= 65.0 and Target_v[1] < 0.0 and Target_v[2] >= 0.0:
                model = model2
                para3 = pd.read_csv('pcmdata0_03.csv')
       

            if Target_v[0] < 45.0 and Target_v[1] < 0.0 and Target_v[2] < 0.0:
                model = model1
                para3 = pd.read_csv('pcmdata0_0_1.csv')


            if Target_v[0] >= 45.0 and Target_v[0] < 65.0 and Target_v[1] < 0.0 and Target_v[2] < 0.0:
                model = model2
                para3 = pd.read_csv('pcmdata0_0_2.csv')


            if Target_v[0] >= 65.0 and Target_v[1] < 0.0 and Target_v[2] < 0.0:
                model = model3
                para3 = pd.read_csv('pcmdata0_0_3.csv')
                
            
            
            
            
            
            with col6:
                
                st.markdown("<h6 style='text-align: left; color: darkblue;'> 2.1. ???????????? ????????????_?????? </h6>", unsafe_allow_html=True)
                
                
                for col in New_x2.columns:
                    New_x2[col] = 0.0
                    for col2 in New_x.columns:
                        if col == col2:
                            New_x2[col] = New_x[col2].values
                            
                            #st.write(col)
        
        
                
                
                
                New_x2.index = ['Old_case']        
                
                st.write("")
                
                st.markdown("<h6 style='text-align: left; color: darkblue;'> ??? ????????? ????????? ?????? </h6>", unsafe_allow_html=True)
                    
                #st.write(New_x2.style.format("{:.5}"))
                
                #scaler = StandardScaler().fit(X_train)
                
                New_x2_new = New_x2.replace(0,np.nan)
                New_x2_new = pd.DataFrame(New_x2_new)
                New_x2_new = New_x2_new.dropna(how='all',axis=1)
                New_x2_new = New_x2_new.replace(np.nan,0)
                st.write(New_x2_new)
                
                
            
                
                predictions = model.predict(New_x2)
                            
                predictions = pd.DataFrame(predictions,columns = ['Pred_L','Pred_a','Pred_b'])
                
                #predictions.index = ['Results'] 
                                
                #st.write(predictions.style.format("{:.5}"))
        
                        
        
                NTarget_v = []
                NTarget_v.append(Target_x['Target_L'] + (predictions['Pred_L']-Actual_x['Target_L']))
                NTarget_v.append(Target_x['Target_a'] + (predictions['Pred_a']-Actual_x['Target_a']))
                NTarget_v.append(Target_x['Target_b'] + (predictions['Pred_b']-Actual_x['Target_b']))
                
                
                
                NTarget_v = pd.DataFrame(NTarget_v)
                NTarget_v = NTarget_v.T
                NTarget_v.columns = Target_n1
                
                Diff = Target_x - NTarget_v
                
                
                #st.write(Diff)
                #st.write(Actual_x)
                #st.write(predictions)
                
                #Diff = Target_v1 - NTarget_v
                
                #st.write(Diff)
                
                df_min0 = New_x2.T
                
                #df_min0 = pd.DataFrame(df_min0)
                #test = df_min0[df_min0 !=0]
                #st.write(test)
                #st.write(df_min0)
                #st.write(df_min0.shape)
                
                #st.write(df_min0)
                #st.write(df_min0.shape())
        
                #df_min0 = pd.DataFrame(df_min0)    
                #st.write(df_min0)
                
                para2 =[]
                para6 = pd.DataFrame()
                
                col_list = []
                for j in range(df_min0.shape[0]):
                        
                    column = df_min0.index[j]
                    
                    #st.write(column)
                    #st.write(df_min0.iloc[j])
                    
                    if df_min0.iloc[j].values > 0 and df_min0.iloc[j].values < 20:
                        
                        min1 = df_min0.iloc[j].values
                        if min1 <0: min1 = 0 
                        
                        max1 = df_min0.iloc[j].values + 5
                        
                        para = np.arange(min1, max1, (max1-min1)/50.0)  
                        col_list.append(column)
                        para2.append(para)
                        
                    if df_min0.iloc[j].values >= 20 and df_min0.iloc[j].values < 40:
                        
                        min1 = df_min0.iloc[j].values
                        if min1 <0: min1 = 0 
                        
                        max1 = df_min0.iloc[j].values + 10
                        
                        para = np.arange(min1, max1, (max1-min1)/50.0)  
                        col_list.append(column)
                        para2.append(para)
                        
                    if df_min0.iloc[j].values >= 40:
                        min1 = df_min0.iloc[j].values
                        
                        if min1 <0: min1 = 0 
                        max1 = df_min0.iloc[j].values + 20
                        
                        para = np.arange(min1, max1, (max1-min1)/50.0)  
                        col_list.append(column)
                        para2.append(para)
                                              
                para2 = pd.DataFrame(para2)
                para2 = para2.T
                #st.write(col_list)
                para6 = para2
                para6.columns = col_list
                
                #st.write(para2)
                #st.write(df_min0)
                #st.write(df_min0.loc['SW1'])
                #st.write(df_min0.iloc[-1])
                
                

                
                
                New_x2 = pd.DataFrame(X.iloc[0,:])
                New_x2 = New_x2.T
                
                       
                
                para7 = []
                random.seed(42)
                for i in range(5000):
                    para5 = []
                    for col in New_x2.columns:
                        New_x2[col] = 0.0
                                           
                        for col1 in list(para6.columns):
                            if col1 == col:
                                New_x2[col] = random.sample(list(para6[col1]),1)
                                    
                                                          
                        para5.append(float(New_x2[col].values))
                      
                    para7.append(para5)
                
                
                para7 = pd.DataFrame(para7,columns=X.columns)
                para7 = para7.replace(np.nan, 0)
                
                #st.write(para7)
                        
                datafile2 = para7.values
                   
                predictions3 = model.predict(datafile2)
           
                predictions3 = pd.DataFrame(predictions3, columns=['Pred_L','Pred_a','Pred_b'])
                   
                para71 = pd.concat([para7,predictions3], axis=1)
                       
                para71 = para71.reset_index(drop = True)
                
                para71['Delta_E'] = 0.0
                                  
                for i in range(para71.shape[0]):
        
                    para71['Delta_E'][i] = ((NTarget_v['Target_L'] - predictions3['Pred_L'][i])**2+(NTarget_v['Target_a'] - predictions3['Pred_a'][i])**2+(NTarget_v['Target_b'] - predictions3['Pred_b'][i])**2)**0.5 
                
                para71.sort_values(by='Delta_E', ascending=True, inplace =True)
                
                para71 = para71.head(3)
                
                #st.write(para71)
                
                
                df_min1 = para71.iloc[0]
                
                
                

                para22 =[]
                para62 = pd.DataFrame()
        
                col_list = []
                for j in range(df_min1.shape[0]-4):
                    
                    column = df_min1.index[j]
                    
               
                    if df_min1.iloc[j] > 0:
                        min1 = float(round(df_min1.iloc[j] ,3))
                        max1 = float(round(df_min1.iloc[j] + 3,3))
                        
                        if min1 < 0 : min1 = 0
                        
                        
                        para = np.arange(min1, max1, (max1-min1)/30.0)  
                        col_list.append(column)
                        para22.append(para)
                                              
          
                para22 = pd.DataFrame(para22)
                para22 = para22.T
                #st.write(col_list)
                para62 = para22
                para62.columns = col_list           
                
                               
        
                
                para72 = []
                random.seed(42)
                for i in range(3000):
                    para5 = []
                    for col in New_x2.columns:
                        New_x2[col] = 0.0
                                           
                        for col1 in list(para62.columns):
                            if col1 == col:
                                New_x2[col] = random.sample(list(para62[col1]),1)
                                    
                                                          
                        para5.append(float(New_x2[col].values))
                      
                    para72.append(para5)
                
                
                para72 = pd.DataFrame(para72,columns=X.columns)
                para72 = para72.replace(np.nan, 0)
                
                #st.write(para72)
                        
                datafile3 = para72.values
                   
                predictions4 = model.predict(datafile3)
           
                predictions4 = pd.DataFrame(predictions4, columns=['Pred_L','Pred_a','Pred_b'])
                   
                para73 = pd.concat([para72,predictions4], axis=1)
                       
                para73 = para73.reset_index(drop = True)
                
                para73['Delta_E'] = 0.0
                                  
                for i in range(para73.shape[0]):
        
                    para73['Delta_E'][i] = ((NTarget_v['Target_L'] - predictions4['Pred_L'][i])**2+(NTarget_v['Target_a'] - predictions4['Pred_a'][i])**2+(NTarget_v['Target_b'] - predictions4['Pred_b'][i])**2)**0.5 
                
                para73.sort_values(by='Delta_E', ascending=True, inplace =True)
                
                para73 = para73.head(1)        
                
                #st.write('1st=', para73)
                st.write('')
                
                st.write('**-????????? ??????**')
                st.write(para73)




                                
                
                
                para_add =[]
                random.seed(42)
                
                for i in range(10):
                    

                    para20 =[]
                    para60 = pd.DataFrame()
            
                    col_list = []
                    col_list2 = []
                    
                    #st.write(df_min0.shape[0])
                    #st.write(df_min0)
                    
                    for j in range(df_min0.shape[0]):
                        
                        column = df_min0.index[j]
                        
                   
                        if df_min0.iloc[j].values > 0:
                            min1 = float(round(df_min0.iloc[j],3))
                            max1 = float(round(df_min0.iloc[j] + 3,3))
                            
                            if min1 < 0 : min1 = 0
                            
                            
                            para = np.arange(min1, max1, (max1-min1)/30.0)  
                            col_list.append(column)
                            para20.append(para)
                            
                            
                        
                        if df_min0.iloc[j].values == 0 :
                            col_list2.append(df_min0.index[j])
                        
                   
                    #st.write(col_list2)
                    
                    col2 = random.sample(list(col_list2),1)
                    
                    col_list.append(col2[0])
    
                    col2_data = np.arange(0,10,0.5)
                    
                    para20.append(col2_data)
                    
                                                  
                    para20 = pd.DataFrame(para20)
                    para20 = para20.T
                    #st.write(col_list)
                    para60 = para20
                    para60.columns = col_list 
                    
                    #st.write(para60)
                    
                    #st.write(df_min0)
                    #st.write(df_min0.loc['SW1'])
                    #st.write(df_min0.iloc[-1])
                    
                    

                    
                    New_x2 = pd.DataFrame(X.iloc[0,:])
                    New_x2 = New_x2.T
                    
                           
                    
                    para70 = []

                    for i in range(2000):
                        para50 = []
                        for col in New_x2.columns:
                            New_x2[col] = 0.0
                                               
                            for col1 in list(para60.columns):
                                if col1 == col:
                                    New_x2[col] = random.sample(list(para60[col1]),1)
                                        
                                                              
                            para50.append(float(New_x2[col].values))
                          
                        para70.append(para50)
                    
                    
                    para70 = pd.DataFrame(para70,columns=X.columns)
                    para70 = para70.replace(np.nan,0)
                    
                    #st.write(para7)
                            
                    datafile20 = para70.values
                    
                       
                    predictions30 = model.predict(datafile20)
               
                    predictions30 = pd.DataFrame(predictions30, columns=['Pred_L','Pred_a','Pred_b'])
                       
                    para710 = pd.concat([para70,predictions30], axis=1)
                           
                    para710 = para710.reset_index(drop = True)
                    
                    para710['Delta_E'] = 0.0
                                      
                    for i in range(para710.shape[0]):
            
                        para710['Delta_E'][i] = ((NTarget_v['Target_L'] - predictions30['Pred_L'][i])**2+(NTarget_v['Target_a'] - predictions30['Pred_a'][i])**2+(NTarget_v['Target_b'] - predictions30['Pred_b'][i])**2)**0.5 
                    
                    para710.sort_values(by='Delta_E', ascending=True, inplace =True)
                    
                    para710 = para710.head(3)
                    
                    #st.write(para710)
                    
                    df_min10 = para710.iloc[0]      

                    para_add.append(df_min10)
                    
                    
                    
                    
                    
                    
                para_add = pd.DataFrame(para_add)
                para_add.sort_values(by='Delta_E', ascending=True, inplace =True)
                
                st.write('')
                st.write('**-?????? ????????? ??????**')
                #st.write(para_add)    

                
                df_min10 = para_add.iloc[0]
                
                
                                   
                para220 =[]
                para620 = pd.DataFrame()
        
                #st.write(df_min10.shape[0])
        
                col_list = []
                
                for j in range(df_min10.shape[0]-4):
                    
                    column = df_min10.index[j]
                    
                    #print(j)
                    
                    if df_min10.iloc[j] > 0:
                        
                        min1 = float(round(df_min10.iloc[j] ,3))
                        max1 = float(round(df_min10.iloc[j] +2,3))
                        

                        para = np.arange(min1, max1, (max1-min1)/50.0)  
                        col_list.append(column)
                        para220.append(para)
                                              
          
                para220 = pd.DataFrame(para220)
                para220 = para220.T
                #st.write(col_list)
                para620 = para220
                para620.columns = col_list           
                
                
                #st.write(para620)

                #st.write(para620)
        
                
                para720 = []
                random.seed(42)
                for i in range(3000):
                    para50 = []
                    for col in New_x2.columns:
                        New_x2[col] = 0.0
                                           
                        for col1 in list(para620.columns):
                            if col1 == col:
                                New_x2[col] = random.sample(list(para620[col1]),1)
                                    
                                                          
                        para50.append(float(New_x2[col].values))
                      
                    para720.append(para50)
                
                
                para720 = pd.DataFrame(para720,columns=X.columns)
                
                para720 = para720.replace(np.nan, 0)
                
                #st.write(para72)
                        
                datafile3 = para720.values
                   
                predictions4 = model.predict(datafile3)
           
                predictions4 = pd.DataFrame(predictions4, columns=['Pred_L','Pred_a','Pred_b'])
                   
                para730 = pd.concat([para720,predictions4], axis=1)
                       
                para730 = para730.reset_index(drop = True)
                
                para730['Delta_E'] = 0.0
                                  
                for i in range(para730.shape[0]):
        
                    para730['Delta_E'][i] = ((NTarget_v['Target_L'] - predictions4['Pred_L'][i])**2+(NTarget_v['Target_a'] - predictions4['Pred_a'][i])**2+(NTarget_v['Target_b'] - predictions4['Pred_b'][i])**2)**0.5 
                
                para730.sort_values(by='Delta_E', ascending=True, inplace =True)
                
                
                para730 = para730.head(1)        
                
                st.write(para730)

                
                
                
                
                para73 = pd.concat([para73,para730], axis=0)
                
                para73.sort_values(by='Delta_E', ascending=True, inplace =True)
                
                #st.write(para73)
                
                
                
                st.write('')
                st.write('')
                st.write('')
                st.write('')
                st.write('')
                
        
                st.markdown("<h6 style='text-align: left; color: darkblue;'> ??? ?????? ????????? ?????? </h6>", unsafe_allow_html=True)
        
        
                para73['Pred_L'] = para73['Pred_L'].values + Diff['Target_L'].values
                para73['Pred_a'] = para73['Pred_a'].values + Diff['Target_a'].values
                para73['Pred_b'] = para73['Pred_b'].values + Diff['Target_b'].values
                    
                
                para73['Delta_L'] = 0.0 
                para73['Delta_a'] = 0.0
                para73['Delta_b'] = 0.0
                para73['Delta_L'] = Target_x['Target_L'].values - para73['Pred_L'].values
                para73['Delta_a'] = Target_x['Target_a'].values - para73['Pred_a'].values
                para73['Delta_b'] = Target_x['Target_b'].values - para73['Pred_b'].values
                
                #st.write(para73)
                
                para73 = para73.drop(['Pred_L','Pred_a','Pred_b'], axis=1)
                

                
                para73_new = para73.replace(0,np.nan)
                para73_new = pd.DataFrame(para73_new)
                para73_new = para73_new.dropna(how='all',axis=1)
                para73_new = para73_new.replace(np.nan,0)
                
                para73_new2 = para73_new.drop(['Delta_E'], axis=1)
                para73_new2['Delta_E'] = para73_new['Delta_E']
                para73_new = para73_new2
                
                
                st.write(para73_new)
            
            
    
