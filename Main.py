import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np

from keras.models import load_model
diabetes_model = pickle.load(open("C:/Users/hp/Disease Pred/Models/diabetes_model.sav", 'rb'))
heart_disease_model = pickle.load(open("C:/Users/hp/Disease Pred/Models/heart_model.sav",'rb'))
parkinsons_model = pickle.load(open("C:/Users/hp/Disease Pred/Models/parkinsons_model.sav", 'rb'))
breast_model = load_model("C:/Users/hp/Disease Pred/Models/breast.h5")
malaria_model = load_model('C:/Users/hp/Disease Pred/Models/cells.h5')

# sidebar for navigation
with st.sidebar:
    selected = option_menu(
            "Multiple Disease Prediction System",
            ["Diabetes Detection", "Heart Disease Detection", "Parkinsons Detection","Breast Cancer Detection","Malaria Detection"], 
            icons=['activity','heart','person'],
            #orientation="horizontal",
        )

# Diabetes Prediction Page
if (selected == 'Diabetes Detection'):
    
    # page title
    st.title('Diabetes Prediction')
    
    
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
        
    with col2:
        Glucose = st.text_input('Glucose Level')
    
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    
    with col2:
        Insulin = st.text_input('Insulin Level')
    
    with col3:
        BMI = st.text_input('BMI value')
    
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    
    with col2:
        Age = st.text_input('Age of the Person')
    
    
    # code for Prediction
    diab_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        
        if (diab_prediction[0] == 1):
            
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'
        
    st.success(diab_diagnosis)
 
# Heart Disease Prediction Page
if (selected == 'Heart Disease Detection'):
    
    # page title
    st.title('Heart Disease Prediction using ML')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age')
        
    with col2:
        sex = st.text_input('Sex')
        
    with col3:
        cp = st.text_input('Chest Pain types')
        
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
        
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
        
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
        
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')
        
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
        
    with col3:
        exang = st.text_input('Exercise Induced Angina')
        
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
        
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')
        
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')
        
    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
        
        
     
     
    # code for Prediction
    heart_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Heart Disease Test Result'):
        heart_prediction = heart_disease_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg,thalach,exang,oldpeak,slope,ca,thal]])                          
        
        if (heart_prediction[0] == 1):
            heart_diagnosis = 'The person is having heart disease'
        else:
            heart_diagnosis = 'The person does not have any heart disease'
        
    st.success(heart_diagnosis)


# Parkinson's Prediction Page
if (selected == "Parkinsons Detection"):
    
    # page title
    st.title("Parkinson's Disease Prediction using ML")
    
    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
        
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
        
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')
        
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')
        
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
        
    with col1:
        RAP = st.text_input('MDVP:RAP')
        
    with col2:
        PPQ = st.text_input('MDVP:PPQ')
        
    with col3:
        DDP = st.text_input('Jitter:DDP')
        
    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')
        
    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
        
    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')
        
    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')
        
    with col3:
        APQ = st.text_input('MDVP:APQ')
        
    with col4:
        DDA = st.text_input('Shimmer:DDA')
        
    with col5:
        NHR = st.text_input('NHR')
        
    with col1:
        HNR = st.text_input('HNR')
        
    with col2:
        RPDE = st.text_input('RPDE')
        
    with col3:
        DFA = st.text_input('DFA')
        
    with col4:
        spread1 = st.text_input('spread1')
        
    with col5:
        spread2 = st.text_input('spread2')
        
    with col1:
        D2 = st.text_input('D2')
        
    with col2:
        PPE = st.text_input('PPE')
        
    
    
    # code for Prediction
    parkinsons_diagnosis = ''
    
    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):
        parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])                          
        
        if (parkinsons_prediction[0] == 1):
            parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
            parkinsons_diagnosis = "The person does not have Parkinson's disease"
        
    st.success(parkinsons_diagnosis)
    
 # Breast Prediction Page
if (selected == "Breast Cancer Detection"):
    # page title
    st.title("Breast Cancer Detection using ML")
    
    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        meanradius = st.text_input('mean radius')
        
    with col2:
        meantexture = st.text_input('mean texture')
        
    with col3:
        meanperimeter = st.text_input('mean perimeter')
        
    with col4:
        meanarea = st.text_input('mean area')
        
    with col5:
        meansmoothness = st.text_input('mean smoothness')
        
    with col1:
        meancompactness = st.text_input('mean compactness')
        
    with col2:
        meanconcavity = st.text_input('mean concavity')
        
    with col3:
        meanconcavepoints = st.text_input('mean concave points')
        
    with col4:
        meansymmetry = st.text_input('mean symmetry')
        
    with col5:
        meanfractaldimension = st.text_input('mean fractal dimension')
        
    with col1:
        radiuserror = st.text_input('radius error')
        
    with col2:
        textureerror = st.text_input('texture error')
        
    with col3:
        perimetererror = st.text_input('perimeter error')
        
    with col4:
        areaerror = st.text_input('area error')
        
    with col5:
        smoothnesserror = st.text_input('smoothness error')
        
    with col1:
        compactnesserror = st.text_input('compactness error')
        
    with col2:
        concavityerror = st.text_input('concavity error')
        
    with col3:
        concavepointserror = st.text_input('concave points error')
        
    with col4:
        symmetryerror = st.text_input('symmetry error')
        
    with col5:
        fractaldimensionerror = st.text_input('fractal dimension error')
        
    with col1:
        worstradius = st.text_input('worst radius')
        
    with col2:
        worsttexture = st.text_input('worst texture')
    
    with col3:
        worstperimeter = st.text_input('worst perimeter')
        
    with col4:
        worstarea = st.text_input('worst area')
        
    with col5:
        worstsmoothness = st.text_input('worst smoothness')
        
    with col1:
        worstcompactness = st.text_input('worst compactness')
                                     
    with col2:
        worstconcavity = st.text_input('worst concavity')
    
    with col3:
        worstconcavepoints = st.text_input('worst concave points')
        
    with col4:
        worstsymmetry = st.text_input('worst symmetry')
        
    with col5:
        worstfractaldimension = st.text_input('worst fractal dimension')
        
    
    
    # code for Prediction
    breast_diagnosis = ''
    
    # creating a button for Prediction    
    if st.button("Breast_diagnosis Result"):
        input_data=(meanradius,meantexture,meanperimeter,meanarea,meansmoothness,meancompactness,meanconcavity,meanconcavepoints,
            meansymmetry,
            meanfractaldimension,
            radiuserror,
            textureerror,
            perimetererror,
            areaerror,
            smoothnesserror,
            compactnesserror,
            concavityerror,
            concavepointserror,
            symmetryerror,
            fractaldimensionerror,
            worstradius,
            worsttexture,
            worstperimeter,
            worstarea,
            worstsmoothness,
            worstcompactness,
            worstconcavity,
            worstconcavepoints,
            worstsymmetry,
            worstfractaldimension)
        input_data_as_numpy_array = np.asarray(input_data)

        # reshape the numpy array as we are predicting for one data point
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

        # standardizing the input data
        input_data_std = scaler.transform(input_data_reshaped)
        breast_prediction = breast_model.predict(input_data_std)  
        prediction_label = [np.argmax(prediction)]

        if(prediction_label[0] == 0):
            breast_diagnosis = "The person has Breast Cancer disease"

        else:
            breast_diagnosis = "The person does not have Breast Cancer  disease"
    
    st.success(breast_diagnosis)
   
   
#for Malaria
if (selected == "Malaria Detection"):
    # page title
    st.title("Malaria Detection using ML")
    from keras.models import load_model
    from PIL import Image
    from PIL import Image
    import numpy as np
    import os
    import cv2
    from io import StringIO

    #adding a file uploader
    def convert_to_array(img):
            im = cv2.imread(img)
            img_ = Image.fromarray(im, 'RGB')
            image = img_.resize((50, 50))
            return np.array(image)

    def get_cell_name(label):
            if label==0:
                return "Paracitized"
            if label==1:
                return "Uninfected"
    def predict_cell(file):
            model = load_model('cells.h5')
            print("Predicting Type of Cell Image.................................")
            ar=convert_to_array(file)
            ar=ar/255
            label=1
            a=[]
            a.append(ar)
            a=np.array(a)
            score=model.predict(a,verbose=1)
            label_index=np.argmax(score)
            print(label_index)
            acc=np.max(score)
            Cell=get_cell_name(label_index)
            return Cell,"The predicted Cell is a "+Cell+" with accuracy =    "+str(acc)

    file = st.file_uploader("Please choose a file")

    if file is not None:

        #To read file as bytes:

        bytes_data = file.getvalue()

        st.write(bytes_data)

        predict_cell(file)



