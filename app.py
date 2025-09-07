import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Konfigurasi halaman
st.set_page_config(
    page_title="BMI Prediction App",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS untuk styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .metric-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 0.5rem 0;
    }
    
    .prediction-result {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Cache untuk loading data dan model
@st.cache_data
def load_data():
    """Load dataset BMI"""
    url = "https://raw.githubusercontent.com/whintaaa/iris/main/500_Person_Gender_Height_Weight_Index.csv"
    return pd.read_csv(url)

@st.cache_resource
def load_models():
    """Load dan train semua model"""
    # Load data
    dt = load_data()
    X = dt.drop('Index', axis=1)
    y = dt['Index']
    
    # Preprocessing
    df = pd.get_dummies(X, prefix='Gender')
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    X_processed = pd.DataFrame(scaled, columns=df.columns)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, train_size=0.8, random_state=1)
    
    # Train models
    models = {}
    
    # KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    models['KNN'] = knn
    
    # Decision Tree
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    models['Decision Tree'] = dt_model
    
    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    models['Naive Bayes'] = nb
    
    # ANN BP
    ann_bp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=500, random_state=42)
    ann_bp.fit(X_train, y_train)
    models['ANN BP'] = ann_bp
    
    return models, scaler, X_test, y_test

# Load models dan data
models, scaler, X_test, y_test = load_models()
data = load_data()

# Label mapping
label_mapping = {
    0: 'Extremely Weak',
    1: 'Weak', 
    2: 'Normal',
    3: 'Overweight',
    4: 'Obesity',
    5: 'Extreme Obesity'
}

def get_bmi_color(index):
    """Return color based on BMI index"""
    colors = {
        0: '#3498db',  # Blue
        1: '#2ecc71',  # Green  
        2: '#27ae60',  # Dark Green
        3: '#f39c12',  # Orange
        4: '#e74c3c',  # Red
        5: '#c0392b'   # Dark Red
    }
    return colors.get(index, '#95a5a6')

# Header
st.markdown('<h1 class="main-header">⚖️ BMI Prediction App</h1>', unsafe_allow_html=True)

# Info author
st.markdown("""
<div class="info-box">
    <h3>🎓 Informasi Aplikasi</h3>
    <p><strong>Nama:</strong> Whinta Virginia Putri</p>
    <p><strong>Deskripsi:</strong> Aplikasi prediksi BMI menggunakan Machine Learning untuk menentukan kategori berat badan berdasarkan tinggi, berat, dan jenis kelamin.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar untuk navigasi
st.sidebar.title("📊 Navigasi")
menu = st.sidebar.selectbox(
    "Pilih Menu:",
    ["🏠 Home", "📈 Data Analysis", "🤖 Model Performance", "🔮 Prediction", "ℹ️ About"]
)

if menu == "🏠 Home":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">Tentang BMI (Body Mass Index)</h2>', unsafe_allow_html=True)
        st.write("""
        **BMI (Body Mass Index)** adalah sebuah metode yang digunakan untuk mengukur proporsi 
        antara berat badan dan tinggi badan seseorang. BMI biasanya digunakan sebagai indikator 
        kasar untuk menentukan apakah seseorang memiliki berat badan yang sehat atau tidak.
        """)
        
        st.markdown("### 📊 Kategori BMI:")
        categories = [
            ("Extremely Weak", "Sangat Kurus", "#3498db"),
            ("Weak", "Kurus", "#2ecc71"), 
            ("Normal", "Normal", "#27ae60"),
            ("Overweight", "Kelebihan Berat", "#f39c12"),
            ("Obesity", "Obesitas", "#e74c3c"),
            ("Extreme Obesity", "Obesitas Ekstrem", "#c0392b")
        ]
        
        for i, (eng, ind, color) in enumerate(categories):
            st.markdown(f"""
            <div style="background-color: {color}; color: white; padding: 0.5rem; 
                       border-radius: 5px; margin: 0.2rem 0;">
                <strong>{i}: {eng}</strong> - {ind}
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📈 Dataset Overview")
        st.metric("Total Data", len(data))
        st.metric("Features", 3)
        st.metric("Classes", 6)
        
        # Distribution chart
        fig = px.histogram(data, x='Index', title="Distribusi Kategori BMI")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

elif menu == "📈 Data Analysis":
    st.markdown('<h2 class="sub-header">📊 Analisis Data</h2>', unsafe_allow_html=True)
    
    # Show dataset
    st.subheader("📋 Dataset BMI")
    st.dataframe(data.head(10), use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gender distribution
        gender_dist = data['Gender'].value_counts()
        fig1 = px.pie(values=gender_dist.values, names=gender_dist.index, 
                     title="Distribusi Gender")
        st.plotly_chart(fig1, use_container_width=True)
        
        # Height vs Weight scatter
        fig3 = px.scatter(data, x='Height', y='Weight', color='Index',
                         title="Height vs Weight by BMI Category",
                         hover_data=['Gender'])
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        # BMI index distribution
        index_dist = data['Index'].value_counts().sort_index()
        fig2 = px.bar(x=index_dist.index, y=index_dist.values,
                     title="Distribusi Kategori BMI")
        fig2.update_traces(marker_color=['#3498db', '#2ecc71', '#27ae60', 
                                        '#f39c12', '#e74c3c', '#c0392b'])
        st.plotly_chart(fig2, use_container_width=True)
        
        # Statistical summary
        st.subheader("📊 Statistik Deskriptif")
        st.dataframe(data.describe(), use_container_width=True)

elif menu == "🤖 Model Performance":
    st.markdown('<h2 class="sub-header">🎯 Performa Model</h2>', unsafe_allow_html=True)
    
    # Calculate accuracies
    accuracies = {}
    predictions = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies[name] = acc
        predictions[name] = y_pred
    
    # Display accuracies
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <h3>KNN</h3>
            <h2 style="color: #667eea;">{accuracies['KNN']:.4f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <h3>Decision Tree</h3>
            <h2 style="color: #667eea;">{accuracies['Decision Tree']:.4f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <h3>Naive Bayes</h3>
            <h2 style="color: #667eea;">{accuracies['Naive Bayes']:.4f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-box">
            <h3>ANN BP</h3>
            <h2 style="color: #667eea;">{accuracies['ANN BP']:.4f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Best model
    best_model = max(accuracies, key=accuracies.get)
    st.success(f"🏆 Model Terbaik: **{best_model}** dengan akurasi **{accuracies[best_model]:.4f}**")
    
    # Accuracy comparison chart
    fig = px.bar(x=list(accuracies.keys()), y=list(accuracies.values()),
                title="Perbandingan Akurasi Model",
                color=list(accuracies.values()),
                color_continuous_scale="Viridis")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

elif menu == "🔮 Prediction":
    st.markdown('<h2 class="sub-header">🔮 Prediksi BMI</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Input Data")
        
        # Input form
        height = st.number_input("Tinggi Badan (cm)", min_value=100.0, max_value=250.0, value=170.0)
        weight = st.number_input("Berat Badan (kg)", min_value=30.0, max_value=200.0, value=70.0)
        gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
        
        # Model selection
        model_choice = st.selectbox("Pilih Model", list(models.keys()))
        
        # Predict button
        if st.button("🔮 Prediksi BMI", type="primary"):
            # Preprocessing
            gender_female = 1 if gender == 'Female' else 0
            gender_male = 1 if gender == 'Male' else 0
            
            input_data = np.array([[height, weight, gender_female, gender_male]])
            input_data_scaled = scaler.transform(input_data)
            
            # Make prediction
            selected_model = models[model_choice]
            prediction = selected_model.predict(input_data_scaled)[0]
            prediction_label = label_mapping[prediction]
            
            # Store in session state
            st.session_state.prediction = prediction
            st.session_state.prediction_label = prediction_label
            st.session_state.model_used = model_choice
    
    with col2:
        st.subheader("📊 Hasil Prediksi")
        
        if 'prediction' in st.session_state:
            color = get_bmi_color(st.session_state.prediction)
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {color} 0%, {color}aa 100%); 
                       color: white; padding: 2rem; border-radius: 15px; 
                       text-align: center; margin: 2rem 0;">
                <h2>Kategori BMI Anda:</h2>
                <h1 style="font-size: 2.5rem; margin: 1rem 0;">
                    {st.session_state.prediction_label}
                </h1>
                <p>Model: {st.session_state.model_used}</p>
                <p>Index: {st.session_state.prediction}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Health recommendations
            recommendations = {
                0: "⚠️ Sangat kurus - Konsultasi dengan dokter untuk program penambahan berat badan yang sehat",
                1: "📈 Kurus - Tingkatkan asupan kalori dengan makanan bergizi",
                2: "✅ Normal - Pertahankan pola hidup sehat!",
                3: "⚖️ Kelebihan berat - Kurangi kalori dan tingkatkan aktivitas fisik",
                4: "🚨 Obesitas - Konsultasi dengan ahli gizi untuk program diet",
                5: "⛔ Obesitas ekstrem - Segera konsultasi dengan dokter"
            }
            
            st.info(recommendations[st.session_state.prediction])
        else:
            st.info("👆 Masukkan data dan klik tombol prediksi untuk melihat hasil")

else:  # About
    st.markdown('<h2 class="sub-header">ℹ️ Tentang Aplikasi</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Tujuan Aplikasi
        Aplikasi ini dibuat untuk memprediksi kategori BMI seseorang berdasarkan:
        - Tinggi badan (cm)
        - Berat badan (kg) 
        - Jenis kelamin
        
        ### 🔬 Model Machine Learning
        - **KNN (K-Nearest Neighbors)**
        - **Decision Tree**
        - **Naive Bayes**
        - **ANN BP (Artificial Neural Network with Backpropagation)**
        
        ### 📊 Dataset
        Dataset berisi 500 data dengan 3 fitur dan 6 kategori BMI.
        Dataset diambil dari Kaggle.
        """)
    
    with col2:
        st.markdown("""
        ### 👨‍💻 Developer
        - **Nama:** Whinta Virginia Putri
        - **NIM:** 210411100047
        - **Kelas:** Pendata B
        - **GitHub:** [Link Repository](https://github.com/whintaaa/streamlit_uas/tree/main)
        
        ### 🛠️ Teknologi
        - **Python**
        - **Streamlit**
        - **Scikit-learn**
        - **Plotly**
        - **Pandas & NumPy**
        
        ### 📈 Preprocessing
        - One-Hot Encoding untuk variabel kategorikal
        - Min-Max Scaling untuk normalisasi
        - Principal Component Analysis (PCA)
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d; padding: 1rem;">
    <p>© 2024 BMI Prediction App | Whinta Virginia Putri | Pendata B</p>
    <p>Built with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True)
