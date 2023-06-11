import streamlit as st
import joblib
import time
import pandas as pd
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# display
st.set_page_config(page_title="WhintaVP", page_icon='icon.png')

@st.cache_data()
def progress():
    with st.spinner('Wait for it...'):
        time.sleep(5)

st.title("UAS PENDATA B")
st.write("Analisis dan Prediksi pada dataset BMI (Body Mass Index).")
st.write('Nama : Whinta Virginia Putri')
st.write('NIM : 210411100047')
st.write('Pendata B')

dataframe, preporcessing, modeling, implementation = st.tabs(
    ["Data BMI", "Prepocessing dan PCA (Principal component analysis)", "Modeling", "Implementation"])

label = ['Extremely Weak', 'Weak', 'Normal', 'Overweight', 'Obesity', 'Extreme Obesity']

with dataframe:
    progress()
    url = "https://www.kaggle.com/datasets/yersever/500-person-gender-height-weight-bodymassindex"
    st.markdown(f'[Dataset BMI]({url})')
    st.write('Height and Weight random generated, Body Mass Index Calculated')
    st.write('BMI (Body Mass Index) adalah sebuah metode yang digunakan untuk mengukur proporsi antara berat badan dan tinggi badan seseorang.')
    st.write('BMI biasanya digunakan sebagai indikator kasar untuk menentukan apakah seseorang memiliki berat badan yang sehat atau tidak.')

    dataset, ket = st.tabs(['Dataset', 'Keterangan Dataset'])
    with ket:
        st.write("""
            Column
            1. Gender: Male / Female
            2. Height: Number(cm)
            3. Weight: Number(Kg)
            4. Index:
            * 0 = Extremely Weak
            * 1 = Weak
            * 2 = Normal
            * 3 = Overweight
            * 4 = Obesity
            * 5 = Extreme Obesity
        """)
    with dataset:
        dt = pd.read_csv('https://raw.githubusercontent.com/whintaaa/iris/main/500_Person_Gender_Height_Weight_Index.csv')
        st.dataframe(dt)
        # pisahkan fitur dan label
        X = dt.drop('Index', axis=1)
        y = dt['Index']

with preporcessing:
    progress()
    scaler = joblib.load("scaled.pkl")
    st.write('One Hot Prepocessing')
    st.write("'One-hot Processing' adalah suatu teknik dalam prapemrosesan data yang digunakan untuk mengubah variabel kategorikal menjadi representasi numerik biner. Tujuannya adalah untuk memungkinkan model atau algoritma pembelajaran mesin untuk mengolah dan memahami variabel kategorikal sebagai fitur dalam analisis atau pemodelan.")
    df = pd.get_dummies(X, prefix='Gender')
    scaled = scaler.fit_transform(df)
    X = pd.DataFrame(scaled, columns=df.columns)
    st.dataframe(X)
    st.write('Principal component analysis')
    st.write('Principal Component Analysis (PCA) adalah suatu metode dalam analisis data yang digunakan untuk mengurangi dimensi dari dataset yang kompleks.')
    st.write('Tujuannya adalah untuk mengidentifikasi pola dan struktur utama (komponen utama) dalam dataset dengan mengubah variabel asli menjadi kombinasi linear baru yang disebut komponen utama.')
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    st.write("Variance Ratio:", pca.explained_variance_ratio_)
    st.write("Principal Components:")
    st.dataframe(pd.DataFrame(X_pca, columns=["PC1", "PC2"]))


with modeling:
    progress()
    # split data
    st.write('Terdapat 4 modeling yaitu KNN, Decision Tree, Naive Bayes, ANN BP')
    st.write('Memakai 3 fitur yaitu Gender, Height, Weight. Sedangkan PCA memakai 2 fitur PC1 dan PC2')
    st.write('berikut label test / target sesungguhnya dan label predict / target prediksi ')
    st.write('target pada kolom index dengan ketentuan: 0 = Extremely Weak, 1 = Weak, 2 = Normal, 3 = Overweight, 4 = Obesity, 5 = Extreme Obesity')
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1)
    knn, dt, nb, annbp, knnpca, nbpca, dtpca, annbppca = st.tabs(["KNeighborsClassifier", "DecisionTreeClassifier", "Naive Bayes", "ANN BP", "KNN PCA", "Naive Bayes PCA", "Decision Tree PCA", "ANN BP PCA"])
    # Melakukan PCA
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    # Simpan model ANN BP
    ann_bp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=500)
    ann_bp.fit(X_train, y_train)

    with open('ann_bp.pkl', 'wb') as f:
        pickle.dump(ann_bp, f)

    # Simpan model Naive Bayes
    naive_bayes = GaussianNB()
    naive_bayes.fit(X_train, y_train)

    with open('naive_bayes.pkl', 'wb') as f:
        pickle.dump(naive_bayes, f)

    with knn:
        progress()
        knn = joblib.load('knn.pkl')
        y_pred_knn = knn.predict(X_test)
        akurasi_knn = accuracy_score(y_test, y_pred_knn)
        label_knn = pd.DataFrame(data={'Label Test': y_test, 'Label Predict': y_pred_knn}).reset_index(drop=True)
        st.success(f'akurasi terhadap data test = {akurasi_knn}')
        st.dataframe(label_knn)
        
    with dt:
        progress()
        d3 = joblib.load('d3.pkl')
        y_pred_d3 = d3.predict(X_test)
        akurasi_d3 = accuracy_score(y_test, y_pred_d3)
        label_d3 = pd.DataFrame(data={'Label Test': y_test, 'Label Predict': y_pred_d3}).reset_index(drop=True)
        st.success(f'akurasi terhadap data test = {akurasi_d3}')
        st.dataframe(label_d3)
    
    with nb:
        progress()
        naive_bayes = joblib.load('naive_bayes.pkl')
        y_pred_nb = naive_bayes.predict(X_test)
        akurasi_nb = accuracy_score(y_test, y_pred_nb)
        label_nb = pd.DataFrame(data={'Label Test': y_test, 'Label Predict': y_pred_nb}).reset_index(drop=True)
        st.success(f'akurasi terhadap data test = {akurasi_nb}')
        st.dataframe(label_nb)
    
    with annbp:
        progress()
        ann_bp = joblib.load('ann_bp.pkl')
        y_pred_annbp = ann_bp.predict(X_test)
        akurasi_annbp = accuracy_score(y_test, y_pred_annbp)
        label_annbp = pd.DataFrame(data={'Label Test': y_test, 'Label Predict': y_pred_annbp}).reset_index(drop=True)
        st.success(f'akurasi terhadap data test = {akurasi_annbp}')
        st.dataframe(label_annbp)
    
    with knnpca:
        progress()
        knn_pca = joblib.load('knn_pca.pkl')
        X_test_pca = pca.transform(X_test)
        y_pred_knnpca = knn_pca.predict(X_test_pca)
        akurasi_knnpca = accuracy_score(y_test, y_pred_knnpca)
        label_knnpca = pd.DataFrame(data={'Label Test': y_test, 'Label Predict': y_pred_knnpca}).reset_index(drop=True)
        st.success(f'akurasi terhadap data test = {akurasi_knnpca}')
        st.dataframe(label_knnpca)

    with nbpca:
        progress()
        naive_bayes_pca = joblib.load('naive_bayes_pca.pkl')
        y_pred_nbpca = naive_bayes_pca.predict(X_test_pca)
        akurasi_nbpca = accuracy_score(y_test, y_pred_nbpca)
        label_nbpca = pd.DataFrame(data={'Label Test': y_test, 'Label Predict': y_pred_nbpca}).reset_index(drop=True)
        st.success(f'akurasi terhadap data test = {akurasi_nbpca}')
        st.dataframe(label_nbpca)

    with dtpca:
        progress()
        d3_pca = joblib.load('d3_pca.pkl')
        y_pred_dtpca = d3_pca.predict(X_test_pca)
        akurasi_dtpca = accuracy_score(y_test, y_pred_dtpca)
        label_dtpca = pd.DataFrame(data={'Label Test': y_test, 'Label Predict': y_pred_dtpca}).reset_index(drop=True)
        st.success(f'akurasi terhadap data test = {akurasi_dtpca}')
        st.dataframe(label_dtpca)

    with annbppca:
        progress()
        ann_bp_pca = joblib.load('ann_bp_pca.pkl')
        y_pred_annbppca = ann_bp_pca.predict(X_test_pca)
        akurasi_annbppca = accuracy_score(y_test, y_pred_annbppca)
        label_annbppca = pd.DataFrame(data={'Label Test': y_test, 'Label Predict': y_pred_annbppca}).reset_index(drop=True)
        st.success(f'akurasi terhadap data test = {akurasi_annbppca}')
        st.dataframe(label_annbppca)


with implementation:
    # height
    height = st.number_input('Tinggi', value=174)
    # weight
    weight = st.number_input('Berat', value=96)
    # gender
    gender = st.selectbox('Jenis Kelamin', ['Laki-Laki', 'Perempuan'])
    gender_female = 1 if gender == 'Perempuan' else 0
    gender_male = 1 if gender == 'Laki-Laki' else 0

    # preprocessing
    input_data = pd.DataFrame([[height, weight, gender_female, gender_male]], columns=X.columns)
    input_data_scaled = scaler.transform(input_data)

    # PCA
    input_data_pca = pca.transform(input_data_scaled)

    # model selection
    model_selected = st.selectbox('Pilih Model', ['KNN', 'Naive Bayes', 'Decision Tree', 'ANN BP'])

    if st.button('Predict'):
        if model_selected == 'KNN':
            pred = knn.predict(input_data_scaled)[0]
            pred_pca = knn_pca.predict(input_data_pca)[0]
        elif model_selected == 'Naive Bayes':
            pred = naive_bayes.predict(input_data_scaled)[0]
            pred_pca = naive_bayes_pca.predict(input_data_pca)[0]
        elif model_selected == 'Decision Tree':
            pred = d3.predict(input_data_scaled)[0]
            pred_pca = d3_pca.predict(input_data_pca)[0]
        elif model_selected == 'ANN BP':
            pred = ann_bp.predict(input_data_scaled)[0]
            pred_pca = ann_bp_pca.predict(input_data_pca)[0]

        label_pred = label[pred]
        label_pred_pca = label[pred_pca]

        st.write('---')
        st.write('Hasil Prediksi')
        st.write('---')

        st.write(f'Prediksi BMI dengan {model_selected} (Tanpa PCA)')
        st.write(f'Prediksi BMI: {label_pred}')
        st.write('---')

        st.write(f'Prediksi BMI dengan {model_selected} (Dengan PCA)')
        st.write(f'Prediksi BMI: {label_pred_pca}')
        st.write('---')


