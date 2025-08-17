
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
import warnings
import io
from datetime import datetime
from fpdf import FPDF

warnings.filterwarnings('ignore')

#Page Configuration (must be first Streamlit command) 
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded"
)

#Login State
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

#Hardcoded credentials (demo only)
VALID_USERNAME = "group1"
VALID_PASSWORD = "machinelearning"

# Login Page
if not st.session_state.logged_in:
    st.title(" Login to Churn Prediction App")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == VALID_USERNAME and password == VALID_PASSWORD:
            st.session_state.logged_in = True
            st.success("✅ Login successful! Loading dashboard...")
            st.experimental_rerun()
        else:
            st.error("❌ Invalid username or password.")
    st.stop()  # Prevent the rest of the app from loading

#Main App Starts After Login
#st.title("📊 Customer Churn Analysis Dashboard")
st.write("Welcome to the churn analysis dashboard! Use the sidebar to navigate.")

# Custom CSS for styling
st.markdown("""<style>
/* Import Google Font */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&display=swap');

/* Global typography */
html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif !important;
    background: linear-gradient(135deg, #f8fafc, #eef2f7) !important;
    color: #1f2937 !important;
}

/* Title */
h1 {
    font-size: 2.2rem !important;
    font-weight: 600 !important;
    color: #0f172a !important;
    margin-bottom: 1rem !important;
}

/* Subheader */
h3 {
    font-size: 1.4rem !important;
    font-weight: 500 !important;
    color: #334155 !important;
    border-left: 4px solid rgba(99,102,241,0.5) !important;
    padding-left: 0.5rem !important;
}

/* Glassmorphism Card Style */
.glass-card {
    background: rgba(255, 255, 255, 0.6) !important;
    border-radius: 18px !important;
    border: 1px solid rgba(255,255,255,0.3) !important;
    box-shadow: 0 4px 30px rgba(0,0,0,0.05) !important;
    backdrop-filter: blur(12px) !important;
}

/* Metric cards */
[data-testid="stMetric"], .stMetric {
    background: rgba(255,255,255,0.5) !important;
    border-radius: 18px !important;
    padding: 1rem !important;
    border: 1px solid rgba(255,255,255,0.3) !important;
    box-shadow: 0px 3px 8px rgba(0,0,0,0.04) !important;
    backdrop-filter: blur(8px) !important;
}
[data-testid="stMetric"] * {
    color: #0f172a !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #3b82f6) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.6rem 1.2rem !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    filter: brightness(1.05) !important;
    box-shadow: 0px 4px 12px rgba(124, 58, 237, 0.3) !important;
}

/* Sidebar with Glass Effect */
[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.5) !important;
    backdrop-filter: blur(10px) !important;
    border-right: 1px solid rgba(255,255,255,0.2) !important;
}
[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3 {
    color: #0f172a !important;
}

/* Tables */
table {
    border-collapse: collapse !important;
    width: 100% !important;
}
thead th {
    background-color: rgba(243,244,246,0.8) !important;
    color: #374151 !important;
    font-weight: 500 !important;
    padding: 0.75rem !important;
    backdrop-filter: blur(6px) !important;
}
tbody tr {
    border-bottom: 1px solid #e5e7eb !important;
}
tbody tr:hover {
    background-color: rgba(249,250,251,0.6) !important;
}

/* Gradient Glass Top Navigation Bar */
header {
    background: linear-gradient(135deg, rgba(124,58,237,0.8), rgba(59,130,246,0.8)) !important;
    color: white !important;
    padding: 0.8rem 2rem !important;
    font-size: 1.1rem !important;
    font-weight: 500 !important;
    border-bottom: none !important;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.05) !important;
    backdrop-filter: blur(14px) !important;
}

/* Alerts */
.stAlert-success {
    background-color: rgba(236,253,245,0.8) !important;
    color: #065f46 !important;
    border-radius: 12px !important;
    border: 1px solid rgba(167,243,208,0.6) !important;
    backdrop-filter: blur(8px) !important;
}
.stAlert-warning {
    background-color: rgba(255,251,235,0.8) !important;
    color: #92400e !important;
    border-radius: 12px !important;
    border: 1px solid rgba(252,211,77,0.6) !important;
    backdrop-filter: blur(8px) !important;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>""", unsafe_allow_html=True)





# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'feature_columns' not in st.session_state:
    st.session_state.feature_columns = []
if 'encoders' not in st.session_state:
    st.session_state.encoders = {}

# Sidebar navigation
st.sidebar.title("🎯 Navigation")
page = st.sidebar.selectbox(
    "Choose a page",
    [" Data Import & Overview", " Data Preprocessing", " Model Training", 
     " Model Evaluation", " Prediction Page", " Interpretation & Report"]
)

# Helper functions
# ---------------------------------------------------
# Function: Create Pie Chart
# ---------------------------------------------------
def create_pie_chart(data, column):
    """
    Creates a pie chart for a categorical column.
    Each slice shows the proportion of each category.
    """
    fig = px.pie(
        # Count the values for each category
        values=data[column].value_counts().values,
        names=data[column].value_counts().index,
        title=f"Distribution of {column}",
        
        # Custom color palette (green shades for consistency)
        color_discrete_sequence=['#73946B', '#9EBC8A', '#DDEB9D', '#F8ED8C'],
        
        # Add a hole in the center → makes it look like a donut chart
        hole=0.3
    )
    
    # Place percentage and label text inside each slice
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    return fig


# ---------------------------------------------------
# Function: Create Bar Chart
# ---------------------------------------------------
def create_bar_chart(data, column):
    """
    Creates a bar chart for a categorical column.
    Each bar represents the count of a category.
    """
    # Count the occurrences of each unique category
    value_counts = data[column].value_counts()
    
    fig = px.bar(
        x=value_counts.index,   # categories on X-axis
        y=value_counts.values,  # counts on Y-axis
        title=f"Distribution of {column}",
        
        # Axis labels
        labels={'x': column, 'y': 'Count'},
        
        # Color bars based on count size, using a gradient color scale
        color=value_counts.values,
        color_continuous_scale='viridis'
    )
    
    # Show the count numbers above each bar
    fig.update_traces(texttemplate='%{y}', textposition='outside')
    
    return fig


# ---------------------------------------------------
# Function: Create Correlation Heatmap
# ---------------------------------------------------
def create_correlation_heatmap(data):
    """
    Creates a heatmap to show correlations between numerical columns.
    Correlation ranges from -1 to +1:
      • +1 = strong positive relationship
      • -1 = strong negative relationship
      • 0 = no relationship
    """
    # Compute correlation matrix (numerical values only)
    corr_matrix = data.corr()
    
    # Display heatmap with color intensity for correlation values
    fig = px.imshow(
        corr_matrix,
        title="Correlation Heatmap",
        color_continuous_scale='RdBu_r',  # Red-Blue color scale
        aspect="auto",                     # Auto adjust cell aspect ratio
        text_auto=True                     # Show correlation numbers in cells
    )
    
    # Set chart size
    fig.update_layout(height=600, width=800)
    
    return fig


# ---------------------------------------------------
# Function: Preprocess Input Data
# ---------------------------------------------------
def preprocess_input(input_data, original_data):
    """
    Preprocess input data for prediction.
    - This is where we clean and transform new user input
    - It should match the same preprocessing steps applied to the training dataset
    """
    # Make a copy of the input data so we don’t modify the original directly
    processed_input = input_data.copy()
    


    
    # Convert TotalCharges to numeric if exists
    if 'TotalCharges' in processed_input.columns:
        processed_input['TotalCharges'] = pd.to_numeric(processed_input['TotalCharges'], errors='coerce')
        processed_input['TotalCharges'] = processed_input['TotalCharges'].fillna(0)
    
    # Get categorical columns (excluding customerID and Churn)
    cat_cols = original_data.select_dtypes(include=['object']).columns.tolist()
    if 'customerID' in cat_cols:
        cat_cols.remove('customerID')
    if 'Churn' in cat_cols:
        cat_cols.remove('Churn')
    
    # Binary and multi-categorical columns
    binary_cols = []
    multi_cat_cols = []
    
    for col in cat_cols:
        if col in processed_input.columns:
            unique_vals = original_data[col].nunique()
            if unique_vals == 2:
                binary_cols.append(col)
            else:
                multi_cat_cols.append(col)
    
    # Apply label encoding for binary columns
    for col in binary_cols:
        if col in st.session_state.encoders:
            # Use stored encoder
            encoder = st.session_state.encoders[col]
            try:
                processed_input[col] = encoder.transform(processed_input[col].astype(str))
            except ValueError:
                # Handle unseen values
                processed_input[col] = 0
        else:
            # Create new encoder
            le = LabelEncoder()
            le.fit(original_data[col].astype(str))
            processed_input[col] = le.transform(processed_input[col].astype(str))
            st.session_state.encoders[col] = le
    
    # Apply one-hot encoding for multi-categorical columns
    if multi_cat_cols:
        processed_input = pd.get_dummies(processed_input, columns=multi_cat_cols, drop_first=True)
    
    # Ensure all feature columns are present
    for col in st.session_state.feature_columns:
        if col not in processed_input.columns:
            processed_input[col] = 0
    
    # Select only feature columns in correct order
    processed_input = processed_input.reindex(columns=st.session_state.feature_columns, fill_value=0)
    
    # Apply scaling if scaler exists
    if st.session_state.scaler is not None:
        numerical_features = original_data.select_dtypes(include=[np.number]).columns.tolist()
        numerical_features = [col for col in numerical_features if col in processed_input.columns]
        if numerical_features:
            processed_input[numerical_features] = st.session_state.scaler.transform(processed_input[numerical_features])
    
    return processed_input

# Page 1: Data Import & Overview
if page == " Data Import & Overview":
    st.markdown('<h1 class="main-header">📊 Customer Churn Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">Data Import & Overview</h2>', unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'], help="Upload a CSV file containing customer data")
    
    # Sample data option
    if st.button("📥 Use Sample Telco Dataset"):
        try:
            np.random.seed(42)
            n_samples = 1000
            
            sample_data = {
                'customerID': [f'C{i:04d}' for i in range(n_samples)],
                'gender': np.random.choice(['Male', 'Female'], n_samples),
                'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
                'Partner': np.random.choice(['Yes', 'No'], n_samples),
                'Dependents': np.random.choice(['Yes', 'No'], n_samples),
                'tenure': np.random.randint(1, 73, n_samples),
                'PhoneService': np.random.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1]),
                'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
                'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
                'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
                'PaymentMethod': np.random.choice([
                    'Electronic check', 'Mailed check', 
                    'Bank transfer (automatic)', 'Credit card (automatic)'
                ], n_samples),
                'MonthlyCharges': np.random.normal(65, 20, n_samples),
                'TotalCharges': np.random.normal(2000, 1500, n_samples),
            }
            
            # Create churn
            churn_prob = (
                (sample_data['tenure'] < 12) * 0.3 +
                (sample_data['MonthlyCharges'] > 80) * 0.2 +
                (np.array(sample_data['Contract']) == 'Month-to-month') * 0.4
            )
            sample_data['Churn'] = np.random.binomial(1, churn_prob, n_samples)
            sample_data['Churn'] = ['Yes' if x == 1 else 'No' for x in sample_data['Churn']]
            
            data = pd.DataFrame(sample_data)
            data['MonthlyCharges'] = np.abs(data['MonthlyCharges'])
            data['TotalCharges'] = np.abs(data['TotalCharges'])
            
            # ✅ Fix TotalCharges: convert to float and replace NaN with 0
            data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce').fillna(0.0)
            
            st.session_state.data = data
            st.success("✅ Sample dataset loaded successfully!")
        
        except Exception as e:
            st.error(f"Error creating sample data: {str(e)}")
    
    # Uploaded data
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            
            # ✅ Fix TotalCharges in uploaded file
            if 'TotalCharges' in data.columns:
                data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce').fillna(0.0)
            
            st.session_state.data = data
            st.success("✅ Data uploaded successfully!")
        
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
    
    # If data is available
    if st.session_state.data is not None:
        data = st.session_state.data
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", f"{data.shape[0]:,}")
        with col2:
            st.metric("Total Columns", data.shape[1])
        with col3:
            st.metric("Missing Values", f"{data.isnull().sum().sum():,}")
        with col4:
            st.metric("Duplicated Rows", f"{data.duplicated().sum():,}")
        
        # Data preview
        st.subheader("📊 Data Preview")
        st.dataframe(data.head(10), use_container_width=True)
        
        # Data info
        with st.expander("🔍 Dataset Information", expanded=False):
            buffer = io.StringIO()
            data.info(buf=buffer)
            st.text(buffer.getvalue())
        
        # Descriptive statistics
        with st.expander("📈 Descriptive Statistics", expanded=False):
            st.dataframe(data.describe(), use_container_width=True)
        
        # Unique values
        with st.expander("🔢 Unique Values Analysis", expanded=False):
            unique_info = [
                {
                    'Column': col,
                    'Unique Values': data[col].nunique(),
                    'Data Type': str(data[col].dtype),
                    'Sample Values': ', '.join(map(str, data[col].unique()[:5]))
                }
                for col in data.columns
            ]
            st.dataframe(pd.DataFrame(unique_info), use_container_width=True)
        
        # ✅ Missing values by column
        with st.expander("🚨 Missing Values Analysis", expanded=False):
            missing_info = data.isnull().sum().reset_index()
            missing_info.columns = ["Column", "Missing Values"]
            missing_info["% Missing"] = (missing_info["Missing Values"] / len(data)) * 100
            missing_info = missing_info[missing_info["Missing Values"] > 0]
            if not missing_info.empty:
                st.dataframe(missing_info, use_container_width=True)
            else:
                st.success("✅ No missing values found in the dataset.")
        
        # Target variable distribution
        if 'Churn' in data.columns:
            st.subheader("🎯 Target Variable Distribution")
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(create_pie_chart(data, 'Churn'), use_container_width=True)
            with col2:
                st.plotly_chart(create_bar_chart(data, 'Churn'), use_container_width=True)
            
            churn_rate = (data['Churn'] == 'Yes').mean() * 100
            total_customers = len(data)
            churned_customers = (data['Churn'] == 'Yes').sum()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Overall Churn Rate", f"{churn_rate:.2f}%")
            with col2:
                st.metric("Churned Customers", f"{churned_customers:,}")
            with col3:
                st.metric("Retained Customers", f"{total_customers - churned_customers:,}")
        
        # Quick insights
        st.subheader("💡 Quick Insights")
        insights = []
        if 'tenure' in data.columns:
            insights.append(f"• Average customer tenure: {data['tenure'].mean():.1f} months")
        if 'MonthlyCharges' in data.columns:
            insights.append(f"• Average monthly charges: ${data['MonthlyCharges'].mean():.2f}")
        if 'TotalCharges' in data.columns:
            insights.append(f"• Average total charges: ${data['TotalCharges'].mean():.2f}")
        
        for insight in insights:
            st.write(insight)
    
    else:
        st.info("👆 Please upload a CSV file or use the sample dataset to begin the analysis.")


# Page 2: Data Preprocessing
elif page == " Data Preprocessing":
    # Section header for preprocessing
    st.markdown('<h2 class="section-header"> Data Preprocessing</h2>', unsafe_allow_html=True)
    
    # Check if data is uploaded from Page 1
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first in the 'Data Import & Overview' page.")
    else:
        # Make a copy of the dataset (so we don’t change the original directly)
        data = st.session_state.data.copy()
        
        # -----------------------------
        # Missing Values Handling
        # -----------------------------
        st.subheader(" Missing Values Treatment")
        
        # Count missing values in each column
        missing_summary = data.isnull().sum()
        if missing_summary.sum() > 0:
            st.write("Missing values found:")
            # Show only the columns with missing values
            st.dataframe(missing_summary[missing_summary > 0].to_frame('Missing Count'))
            
            # Special handling for "TotalCharges" column (common in churn dataset)
            if 'TotalCharges' in data.columns:
                # Convert to numeric (non-numeric values become NaN)
                data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
                
                # Count how many missing values remain in TotalCharges
                missing_total_charges = data['TotalCharges'].isnull().sum()
                
                # If missing values exist, let user decide how to fill them
                if missing_total_charges > 0:
                    st.warning(f"Found {missing_total_charges} missing values in TotalCharges")
                    fill_method = st.selectbox(
                        "Choose method to fill missing TotalCharges:",
                        ["Fill with 0", "Fill with median", "Fill with mean", "Drop rows"]
                    )
                    
                    # Apply the chosen method
                    if fill_method == "Fill with 0":
                        data['TotalCharges'] = data['TotalCharges'].fillna(0)
                    elif fill_method == "Fill with median":
                        data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].median())
                    elif fill_method == "Fill with mean":
                        data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].mean())
                    else:
                        # Drop rows that have missing TotalCharges
                        data = data.dropna(subset=['TotalCharges'])
                    
                    st.success(f"✅ Missing values handled using: {fill_method}")
        else:
            st.success("✅ No missing values found!")
        
        # -----------------------------
        # Feature Engineering
        # -----------------------------
        st.subheader("🛠️ Feature Engineering")
        
        # Identify categorical and numerical columns
        cat_cols = data.select_dtypes(include=['object']).columns.tolist()
        if 'customerID' in cat_cols:  # drop customerID since it's just an identifier
            cat_cols.remove('customerID')
        
        num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Show the detected categorical and numerical columns
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Categorical Columns:**")
            for col in cat_cols:
                st.write(f"• {col}: {data[col].nunique()} unique values")
        with col2:
            st.write("**Numerical Columns (updated):**")
            for col in num_cols:
                st.write(f"• {col}: {data[col].dtype}")
        
        # -----------------------------
        # Feature Distributions
        # -----------------------------
        st.subheader("📊 Feature Distributions")
        
        # ---- Numerical Columns ----
        if num_cols:
            st.write("### Histograms with KDE")
            
            # Let user pick which numerical columns to visualize
            selected_num_cols = st.multiselect(
                "Select numerical columns to visualize:",
                num_cols,
                default=num_cols[:3] if len(num_cols) > 3 else num_cols
            )
            
            import plotly.figure_factory as ff
            for col in selected_num_cols:
                # Create distribution plot (histogram + KDE curve)
                hist_data = [data[col].dropna()]  # drop missing values before plotting
                fig_hist = ff.create_distplot(
                    hist_data, [col],
                    show_hist=True, show_rug=False, colors=['#73946B']
                )
                fig_hist.update_layout(title_text=f"Distribution with KDE for {col}")
                st.plotly_chart(fig_hist, use_container_width=True)
        
        # ---- Categorical Columns ----
        if cat_cols:
            st.write("### Categorical Features")
            
            # Let user pick which categorical columns to visualize
            selected_cat_cols = st.multiselect(
                "Select categorical columns to visualize:",
                cat_cols,
                default=cat_cols[:3] if len(cat_cols) > 3 else cat_cols
            )
            
            for col in selected_cat_cols:
                col1, col2 = st.columns(2)
                
                # Pie Chart
                with col1:
                    fig_pie = px.pie(
                        data, names=col,
                        title=f"Pie Chart of {col}",
                        hole=0.3
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                # Bar Chart (fix index issue by renaming columns)
                with col2:
                    cat_counts = data[col].value_counts().reset_index()
                    cat_counts.columns = [col, "count"]  # rename columns properly
                    
                    fig_bar = px.bar(
                        cat_counts,
                        x=col, y="count",
                        title=f"Bar Chart of {col}",
                        labels={col: col, "count": "Count"},
                        color_discrete_sequence=['#73946B']
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

        
        # Encoding options
        st.subheader("🔤 Categorical Encoding")
        
        # Prepare data for modeling
        df_processed = data.copy()
        
        # Drop customerID if exists
        if 'customerID' in df_processed.columns:
            df_processed = df_processed.drop(columns=['customerID'])
        
        # Encode target variable
        if 'Churn' in df_processed.columns:
            df_processed['Churn'] = df_processed['Churn'].map({'No': 0, 'Yes': 1})
            st.success("✅ Target variable encoded (No: 0, Yes: 1)")
        
# Binary categorical columns
# Ensure TotalCharges is numeric
        if "TotalCharges" in df_processed.columns:
            df_processed["TotalCharges"] = pd.to_numeric(df_processed["TotalCharges"], errors="coerce")

        # Automatically exclude high-cardinality columns (more than 10 unique values)
        binary_cols = []
        multi_cat_cols = []

        for col in cat_cols:
            if col != 'Churn' and col in df_processed.columns:
                unique_vals = df_processed[col].nunique()

                if unique_vals <= 10:  # Only encode if 10 or fewer unique values
                    if unique_vals == 2:
                        binary_cols.append(col)
                    else:
                        multi_cat_cols.append(col)
                else:
                    st.write(f"⚠️ Skipping column '{col}' (has {unique_vals} unique values, treated as numeric).")

        if binary_cols:
            st.write(f"**Binary columns to be label encoded:** {binary_cols}")
        if multi_cat_cols:
            st.write(f"**Multi-category columns to be one-hot encoded:** {multi_cat_cols}")

        # Label encoding for binary columns
        encoders = {}
        for col in binary_cols:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            encoders[col] = le

        # One-hot encoding for multi-category columns
        if multi_cat_cols:
            df_processed = pd.get_dummies(df_processed, columns=multi_cat_cols, drop_first=True)
            st.success(f"✅ One-hot encoding applied to {len(multi_cat_cols)} columns")

        # Store encoders
        st.session_state.encoders = encoders

        # Feature scaling
        st.subheader("⚖️ Feature Scaling")

        
        numerical_features = [col for col in num_cols if col in df_processed.columns]
        
        if numerical_features:
            scale_method = st.selectbox(
                "Choose scaling method:",
                ["StandardScaler (Z-score)", "No Scaling"]
            )
            
            if scale_method == "StandardScaler (Z-score)":
                scaler = StandardScaler()
                df_processed[numerical_features] = scaler.fit_transform(df_processed[numerical_features])
                st.session_state.scaler = scaler
                st.success("✅ Features scaled using StandardScaler")
                
                # Show scaling statistics
                with st.expander("Scaling Statistics"):
                    scaling_stats = pd.DataFrame({
                        'Feature': numerical_features,
                        'Mean': scaler.mean_,
                        'Std': scaler.scale_
                    })
                    st.dataframe(scaling_stats)
        
        # Store processed data
        st.session_state.processed_data = df_processed
        st.session_state.feature_columns = [col for col in df_processed.columns if col != 'Churn']
        
        # Show processed data preview
        st.subheader(" Processed Data Preview")
        st.dataframe(df_processed.head(), use_container_width=True)
        
        # Dataset shape comparison
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Original Shape", f"{data.shape[0]} × {data.shape[1]}")
        with col2:
            st.metric("Processed Shape", f"{df_processed.shape[0]} × {df_processed.shape[1]}")
        
        # Correlation analysis
        if df_processed.select_dtypes(include=[np.number]).shape[1] > 1:
            st.subheader("🔗 Correlation Analysis")
            numeric_df = df_processed.select_dtypes(include=[np.number])
            
            # Overall correlation heatmap
            with st.expander("Full Correlation Heatmap", expanded=False):
                fig_corr = create_correlation_heatmap(numeric_df)
                st.plotly_chart(fig_corr, use_container_width=True)
            
            # Feature correlation with target
            if 'Churn' in numeric_df.columns:
                st.subheader("🎯 Feature Correlation with Churn")
                churn_corr = numeric_df.corr()['Churn'].sort_values(key=abs, ascending=False)
                churn_corr = churn_corr.drop('Churn')  # Remove self-correlation
                
                # Show top correlations
                fig_churn_corr = px.bar(
                    x=churn_corr.values[:15],
                    y=churn_corr.index[:15],
                    orientation='h',
                    title="Top 15 Features Correlation with Churn",
                    labels={'x': 'Correlation', 'y': 'Features'},
                    color=churn_corr.values[:15],
                    color_continuous_scale='RdBu_r'
                )
                st.plotly_chart(fig_churn_corr, use_container_width=True)
                
                # Correlation insights
                st.write("**Key Insights:**")
                positive_corr = churn_corr[churn_corr > 0.1].head(3)
                negative_corr = churn_corr[churn_corr < -0.1].head(3)
                
                if not positive_corr.empty:
                    st.write("Strongly positive correlations with churn:")
                    for feature, corr in positive_corr.items():
                        st.write(f"• {feature}: {corr:.3f}")
                
                if not negative_corr.empty:
                    st.write("Strongly negative correlations with churn:")
                    for feature, corr in negative_corr.items():
                        st.write(f"• {feature}: {corr:.3f}")

# Page 3: Model Training
elif page == " Model Training":
    st.markdown('<h2 class="section-header"> Model Training</h2>', unsafe_allow_html=True)
    
    if st.session_state.processed_data is None:
        st.warning("⚠️ Please complete data preprocessing first.")
    else:
        processed_data = st.session_state.processed_data
        
        # Prepare features and target
        if 'Churn' not in processed_data.columns:
            st.error("Target variable 'Churn' not found in processed data.")
        else:
            X = processed_data.drop('Churn', axis=1)
            y = processed_data['Churn']
            
            # Train-test split
            st.subheader("🔄 Train-Test Split Configuration")
            
            col1, col2 = st.columns(2)
            with col1:
                test_size = st.slider("Test Size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
            with col2:
                random_state = st.number_input("Random State", value=42, min_value=0)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # Display split statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Training Samples", f"{X_train.shape[0]:,}")
            with col2:
                st.metric("Testing Samples", f"{X_test.shape[0]:,}")
            with col3:
                st.metric("Features", X_train.shape[1])
            with col4:
                st.metric("Train Churn Rate", f"{y_train.mean():.2%}")
            
            # Model selection and training
            st.subheader("🎯 Model Selection & Training")
            
            models_to_train = st.multiselect(
                "Select models to train:",
                ["Logistic Regression", "Neural Network (MLP)"],
                default=["Logistic Regression", "Neural Network (MLP)"]
            )
            
            # Model hyperparameters
            with st.expander("⚙️ Model Hyperparameters", expanded=False):
                if "Neural Network (MLP)" in models_to_train:
                    st.write("**Neural Network Parameters:**")
                    mlp_hidden_size = st.slider("Hidden Layer Size", 50, 300, 100)
                    mlp_max_iter = st.slider("Max Iterations", 200, 1000, 500)
            
            if st.button("🚀 Train Models", type="primary"):
                models = {}
                
                with st.spinner("Training models..."):
                    progress_bar = st.progress(0)
                    
                    # Logistic Regression
                    if "Logistic Regression" in models_to_train:
                        st.write("🔄 Training Logistic Regression...")
                        lr_model = LogisticRegression(random_state=random_state, max_iter=1000)
                        lr_model.fit(X_train, y_train)
                        
                        lr_pred = lr_model.predict(X_test)
                        lr_pred_proba = lr_model.predict_proba(X_test)[:, 1]
                        lr_accuracy = accuracy_score(y_test, lr_pred)
                        lr_precision = precision_score(y_test, lr_pred)
                        lr_recall = recall_score(y_test, lr_pred)
                        lr_f1 = f1_score(y_test, lr_pred)
                        lr_roc_auc = roc_auc_score(y_test, lr_pred_proba)
                        
                        models['Logistic Regression'] = {
                            'model': lr_model,
                            'predictions': lr_pred,
                            'probabilities': lr_pred_proba,
                            'accuracy': lr_accuracy,
                            'precision': lr_precision,
                            'recall': lr_recall,
                            'f1': lr_f1,
                            'roc_auc': lr_roc_auc,
                            'X_test': X_test,
                            'y_test': y_test
                        }
                        st.success(f"✅ Logistic Regression - Accuracy: {lr_accuracy:.4f}")
                        progress_bar.progress(1/len(models_to_train))
                    
                    # Neural Network
                    if "Neural Network (MLP)" in models_to_train:
                        st.write("🔄 Training Neural Network...")
                        mlp_model = MLPClassifier(
                            hidden_layer_sizes=(mlp_hidden_size,),
                            max_iter=mlp_max_iter,
                            random_state=random_state,
                            early_stopping=True,
                            validation_fraction=0.1
                        )
                        mlp_model.fit(X_train, y_train)
                        
                        mlp_pred = mlp_model.predict(X_test)
                        mlp_pred_proba = mlp_model.predict_proba(X_test)[:, 1]
                        mlp_accuracy = accuracy_score(y_test, mlp_pred)
                        mlp_precision = precision_score(y_test, mlp_pred)
                        mlp_recall = recall_score(y_test, mlp_pred)
                        mlp_f1 = f1_score(y_test, mlp_pred)
                        mlp_roc_auc = roc_auc_score(y_test, mlp_pred_proba)
                        
                        models['Neural Network'] = {
                            'model': mlp_model,
                            'predictions': mlp_pred,
                            'probabilities': mlp_pred_proba,
                            'accuracy': mlp_accuracy,
                            'precision': mlp_precision,
                            'recall': mlp_recall,
                            'f1': mlp_f1,
                            'roc_auc': mlp_roc_auc,
                            'X_test': X_test,
                            'y_test': y_test
                        }
                        st.success(f"✅ Neural Network - Accuracy: {mlp_accuracy:.4f}")
                        progress_bar.progress(1.0)
                
                # Store models in session state
                st.session_state.models = models
                
                # Display training results
                if models:
                    st.subheader("📊 Training Results Summary")
                    results_df = pd.DataFrame({
                        'Model': list(models.keys()),
                        'Accuracy': [models[model]['accuracy'] for model in models.keys()],
                        'Precision': [models[model]['precision'] for model in models.keys()],
                        'Recall': [models[model]['recall'] for model in models.keys()],
                        'F1-Score': [models[model]['f1'] for model in models.keys()],
                        'ROC-AUC': [models[model]['roc_auc'] for model in models.keys()]
                    })
                    
                    # Style the dataframe
                    styled_df = results_df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'])
                    st.dataframe(styled_df, use_container_width=True)
                    
                    # Visualization of results
                    st.subheader(" Model Performance Comparison")
                    
                    metrics_df = results_df.melt(
                        id_vars=['Model'], 
                        value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                        var_name='Metric', value_name='Score'
                    )
                    
                    fig_results = px.bar(
                        metrics_df,
                        x='Model',
                        y='Score',
                        color='Metric',
                        barmode='group',
                        title="Model Performance Comparison",
                        color_discrete_sequence=px.colors.qualitative.Set2
                    )
                    fig_results.update_layout(height=500)
                    st.plotly_chart(fig_results, use_container_width=True)
                    
                    # Best model recommendation
                    best_model_name = results_df.loc[results_df['ROC-AUC'].idxmax(), 'Model']
                    best_roc_auc = results_df.loc[results_df['ROC-AUC'].idxmax(), 'ROC-AUC']
                    
                    st.success(f"🏆 Best performing model: **{best_model_name}** with ROC-AUC of {best_roc_auc:.4f}")


# Page 4: Model Evaluation
elif page == " Model Evaluation":
    st.markdown('<h2 class="section-header"> Model Evaluation</h2>', unsafe_allow_html=True)
    
    if not st.session_state.models:
        st.warning("⚠️ Please train models first.")
    else:
        models = st.session_state.models
        
        # Model selection for evaluation
        selected_model = st.selectbox(
            "Select model for detailed evaluation:",
            list(models.keys())
        )
        
        if selected_model:
            model_info = models[selected_model]
            model = model_info['model']
            y_pred = model_info['predictions']
            y_pred_proba = model_info['probabilities']
            y_test = model_info['y_test']
            X_test = model_info['X_test']
            
            # Performance metrics
            st.subheader("🎯 Performance Metrics")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Accuracy", f"{model_info['accuracy']:.4f}")
            with col2:
                st.metric("Precision", f"{model_info['precision']:.4f}")
            with col3:
                st.metric("Recall", f"{model_info['recall']:.4f}")
            with col4:
                st.metric("F1-Score", f"{model_info['f1']:.4f}")
            with col5:
                st.metric("ROC-AUC", f"{model_info['roc_auc']:.4f}")
            
            # Confusion Matrix
            st.subheader("🔲 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                title="Confusion Matrix",
                labels=dict(x="Predicted", y="Actual"),
                x=['No Churn', 'Churn'],
                y=['No Churn', 'Churn'],
                color_continuous_scale='Blues'
            )
            fig_cm.update_layout(height=400, width=500)
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.plotly_chart(fig_cm, use_container_width=True)
            
            with col2:
                # Confusion matrix interpretation
                tn, fp, fn, tp = cm.ravel()
                st.write("**Confusion Matrix Breakdown:**")
                st.write(f"• True Negatives (TN): {tn}")
                st.write(f"• False Positives (FP): {fp}")
                st.write(f"• False Negatives (FN): {fn}")
                st.write(f"• True Positives (TP): {tp}")
                
                st.write("**Business Impact:**")
                st.write(f"• Correctly identified non-churners: {tn}")
                st.write(f"• Incorrectly flagged as churners: {fp}")
                st.write(f"• Missed churners: {fn}")
                st.write(f"• Correctly identified churners: {tp}")
            
            # Classification Report
            st.subheader("📊 Detailed Classification Report")
            class_report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(class_report).transpose()
            
            # Format the report for better display
            report_df = report_df.round(4)
            st.dataframe(report_df.style.highlight_max(axis=0), use_container_width=True)
            
            # ROC Curve
            st.subheader(" ROC Curve Analysis")
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
            
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'{selected_model} (AUC = {model_info["roc_auc"]:.4f})',
                line=dict(color='blue', width=2)
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(color='red', dash='dash')
            ))
            fig_roc.update_layout(
                title='ROC Curve',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                height=500
            )
            st.plotly_chart(fig_roc, use_container_width=True)
            
            # Feature Importance
            if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
                st.subheader("🔍 Feature Importance Analysis")
                
                if hasattr(model, 'feature_importances_'):
                    # For Random Forest
                    importance_values = model.feature_importances_
                    importance_type = "Feature Importance"
                elif hasattr(model, 'coef_'):
                    # For Logistic Regression
                    importance_values = abs(model.coef_[0])
                    importance_type = "Absolute Coefficients"
                
                feature_importance = pd.DataFrame({
                    'Feature': X_test.columns,
                    'Importance': importance_values
                }).sort_values('Importance', ascending=False)
                
                # Top 15 features
                top_features = feature_importance.head(15)
                
                fig_importance = px.bar(
                    top_features,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title=f"Top 15 {importance_type}",
                    color='Importance',
                    color_continuous_scale='viridis'
                )
                fig_importance.update_layout(height=600)
                st.plotly_chart(fig_importance, use_container_width=True)
                
                # Feature importance table
                with st.expander(" Complete Feature Importance Table"):
                    st.dataframe(feature_importance, use_container_width=True)
            
            # Prediction Distribution
            st.subheader("📊 Prediction Probability Distribution")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_dist = px.histogram(
                    x=y_pred_proba,
                    nbins=30,
                    title="Distribution of Prediction Probabilities",
                    labels={'x': 'Churn Probability', 'y': 'Count'},
                    color_discrete_sequence=['#73946B']
                )
                st.plotly_chart(fig_dist, use_container_width=True)
            
            with col2:
                # Probability by actual class
                prob_df = pd.DataFrame({
                    'Probability': y_pred_proba,
                    'Actual': y_test.map({0: 'No Churn', 1: 'Churn'})
                })
                
                fig_box_prob = px.box(
                    prob_df, x='Actual', y='Probability',
                    title="Prediction Probability by Actual Class",
                    color='Actual',
                    color_discrete_sequence=['#73946B', '#F8ED8C']
                )
                st.plotly_chart(fig_box_prob, use_container_width=True)
            
            # Model-specific insights
            st.subheader("💡 Model Insights")
            
            if selected_model == "Logistic Regression":
                st.write("""
                **Logistic Regression Insights:**
                - Linear relationship between features and log-odds of churn
                - Coefficients indicate the change in log-odds for unit change in feature
                - Good baseline model with interpretable results
                - Assumes linear relationship between features and target
                """)
            elif selected_model == "Random Forest":
                st.write(f"""
                **Random Forest Insights:**
                - Ensemble of {model.n_estimators} decision trees
                - Feature importance based on decrease in node impurity
                - Handles non-linear relationships and feature interactions
                - Less prone to overfitting compared to single decision tree
                """)
            elif selected_model == "Neural Network":
                st.write(f"""
                **Neural Network Insights:**
                - Multi-layer perceptron with {len(model.hidden_layer_sizes)} hidden layer(s)
                - Can capture complex non-linear patterns
                - Black-box model with less interpretability
                - Converged in {model.n_iter_} iterations
                """)

# Page 5: Prediction Page
elif page == " Prediction Page":
    # Display main section header with custom HTML styling
    st.markdown('<h2 class="section-header"> Customer Churn Prediction</h2>', unsafe_allow_html=True)
    
    # Check if models and processed data exist in session state
    if not st.session_state.models or st.session_state.processed_data is None:
        st.warning("⚠️ Please complete model training first.")
    else:
        st.subheader("🎯 Make Predictions for New Customers")
        
        
        # MODEL SELECTION
        
        # Dropdown to select which trained model to use for predictions
        selected_model = st.selectbox(
            "Select model for prediction:",
            list(st.session_state.models.keys()),  # Get all available model names
            help="Choose the trained model to use for predictions"
        )
        
        
        # INPUT METHOD SELECTION
         
        # Radio buttons to choose between manual input or CSV upload
        input_method = st.radio(
            "Choose input method:",
            ["🖊️ Manual Input", "📁 Upload CSV"],
            horizontal=True
        )
        
        # 
        # MANUAL INPUT METHOD
        # 
        if input_method == "🖊️ Manual Input":
            st.subheader("📝 Enter Customer Information")
            
            # Get original data structure to create appropriate input fields
            original_data = st.session_state.data
            input_data = {}  # Dictionary to store user inputs
            
            # Create organized tabs for different categories of customer information
            tab1, tab2, tab3 = st.tabs(["👤 Demographics", "📞 Services", "💰 Billing"])
            
            # ===== DEMOGRAPHICS TAB =====
            with tab1:
                col1, col2 = st.columns(2)  # Two-column layout
                
                with col1:
                    # Gender selection dropdown
                    if 'gender' in original_data.columns:
                        input_data['gender'] = st.selectbox("Gender", original_data['gender'].unique())
                    
                    # Senior citizen status (0/1 converted to Yes/No display)
                    if 'SeniorCitizen' in original_data.columns:
                        input_data['SeniorCitizen'] = st.selectbox(
                            "Senior Citizen", 
                            [0, 1], 
                            format_func=lambda x: 'Yes' if x == 1 else 'No'
                        )
                    
                    # Partner status selection
                    if 'Partner' in original_data.columns:
                        input_data['Partner'] = st.selectbox("Has Partner", original_data['Partner'].unique())
                
                with col2:
                    # Dependents status selection
                    if 'Dependents' in original_data.columns:
                        input_data['Dependents'] = st.selectbox("Has Dependents", original_data['Dependents'].unique())
                    
                    # Tenure input (months with customer)
                    if 'tenure' in original_data.columns:
                        input_data['tenure'] = st.number_input(
                            "Tenure (months)", 
                            min_value=0, 
                            max_value=100, 
                            value=12
                        )
            
            # SERVICES TAB
            with tab2:
                col1, col2 = st.columns(2)  # Two-column layout
                
                with col1:
                    # Phone service selection
                    if 'PhoneService' in original_data.columns:
                        input_data['PhoneService'] = st.selectbox("Phone Service", original_data['PhoneService'].unique())
                    
                    # Multiple lines service
                    if 'MultipleLines' in original_data.columns:
                        input_data['MultipleLines'] = st.selectbox("Multiple Lines", original_data['MultipleLines'].unique())
                    
                    # Internet service type
                    if 'InternetService' in original_data.columns:
                        input_data['InternetService'] = st.selectbox("Internet Service", original_data['InternetService'].unique())
                    
                    # Online security add-on
                    if 'OnlineSecurity' in original_data.columns:
                        input_data['OnlineSecurity'] = st.selectbox("Online Security", original_data['OnlineSecurity'].unique())
                
                with col2:
                    # Online backup service
                    if 'OnlineBackup' in original_data.columns:
                        input_data['OnlineBackup'] = st.selectbox("Online Backup", original_data['OnlineBackup'].unique())
                    
                    # Device protection service
                    if 'DeviceProtection' in original_data.columns:
                        input_data['DeviceProtection'] = st.selectbox("Device Protection", original_data['DeviceProtection'].unique())
                    
                    # Technical support service
                    if 'TechSupport' in original_data.columns:
                        input_data['TechSupport'] = st.selectbox("Tech Support", original_data['TechSupport'].unique())
                    
                    # Streaming TV service
                    if 'StreamingTV' in original_data.columns:
                        input_data['StreamingTV'] = st.selectbox("Streaming TV", original_data['StreamingTV'].unique())
                    
                    # Streaming movies service
                    if 'StreamingMovies' in original_data.columns:
                        input_data['StreamingMovies'] = st.selectbox("Streaming Movies", original_data['StreamingMovies'].unique())
            
            #BILLING TAB 
            with tab3:
                col1, col2 = st.columns(2)  # Two-column layout
                
                with col1:
                    # Contract type selection
                    if 'Contract' in original_data.columns:
                        input_data['Contract'] = st.selectbox("Contract Type", original_data['Contract'].unique())
                    
                    # Paperless billing preference
                    if 'PaperlessBilling' in original_data.columns:
                        input_data['PaperlessBilling'] = st.selectbox("Paperless Billing", original_data['PaperlessBilling'].unique())
                
                with col2:
                    # Payment method selection
                    if 'PaymentMethod' in original_data.columns:
                        input_data['PaymentMethod'] = st.selectbox("Payment Method", original_data['PaymentMethod'].unique())
                    
                    # Monthly charges input
                    if 'MonthlyCharges' in original_data.columns:
                        input_data['MonthlyCharges'] = st.number_input(
                            "Monthly Charges ($)", 
                            min_value=0.0, 
                            max_value=200.0, 
                            value=65.0
                        )
                    
                    # Total charges input (auto-calculated based on tenure and monthly charges)
                    if 'TotalCharges' in original_data.columns:
                        if 'tenure' in input_data and 'MonthlyCharges' in input_data:
                            # Calculate suggested total based on tenure * monthly charges
                            suggested_total = input_data['tenure'] * input_data['MonthlyCharges']
                            input_data['TotalCharges'] = st.number_input(
                                "Total Charges ($)", 
                                min_value=0.0, 
                                value=float(suggested_total)
                            )
                        else:
                            # Default value if auto-calculation not possible
                            input_data['TotalCharges'] = st.number_input(
                                "Total Charges ($)", 
                                min_value=0.0, 
                                value=1000.0
                            )
            
            
            # PREDICTION EXECUTION
            
            # Primary button to trigger prediction
            if st.button("🎯 Predict Churn Risk", type="primary"):
                try:
                    # Convert input dictionary to DataFrame
                    input_df = pd.DataFrame([input_data])
                    
                    # Preprocess the input to match training data format
                    processed_input = preprocess_input(input_df, original_data)
                    
                    # Get the selected trained model
                    model = st.session_state.models[selected_model]['model']
                    
                    # Make prediction (0 = No Churn, 1 = Churn)
                    prediction = model.predict(processed_input)[0]
                    # Get probability of churn (second element of probability array)
                    probability = model.predict_proba(processed_input)[0][1]
                    
                    
                    # DISPLAY PREDICTION RESULTS
                    
                    st.subheader("🎯 Prediction Results")
                    
                    # Three-column layout for key metrics
                    col1, col2, col3 = st.columns(3)
                    
                    # Column 1: Churn risk status
                    with col1:
                        if prediction == 1:
                            st.error(f"⚠️ **HIGH CHURN RISK**")
                        else:
                            st.success(f"✅ **LOW CHURN RISK**")
                    
                    # Column 2: Churn probability percentage
                    with col2:
                        st.metric("Churn Probability", f"{probability:.1%}")
                    
                    # Column 3: Risk level categorization
                    with col3:
                        risk_level = "High" if probability > 0.7 else "Medium" if probability > 0.3 else "Low"
                        st.metric("Risk Level", risk_level)
                    
                    
                    # PROBABILITY GAUGE VISUALIZATION
                    
                    # Create interactive gauge chart showing churn probability
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = probability * 100,  # Convert to percentage
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Churn Probability (%)"},
                        delta = {'reference': 50},  # Reference line at 50%
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgreen"},    # Low risk: green
                                {'range': [30, 70], 'color': "yellow"},      # Medium risk: yellow
                                {'range': [70, 100], 'color': "red"}         # High risk: red
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 70  # Threshold line at 70%
                            }
                        }
                    ))
                    fig_gauge.update_layout(height=400)
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    
                    # ACTIONABLE RECOMMENDATIONS
                    
                    st.subheader("💡 Recommendations")
                    
                    # Risk-based recommendations
                    if probability > 0.7:  # High risk (>70%)
                        st.write("""
                        **🚨 Immediate Action Required:**
                        - Contact customer within 24 hours
                        - Offer retention incentives (discounts, upgrades)
                        - Schedule a call to understand concerns
                        - Consider contract extension offers
                        """)
                    elif probability > 0.3:  # Medium risk (30-70%)
                        st.write("""
                        **⚠️ Monitor Closely:**
                        - Add to watch list for proactive monitoring
                        - Send satisfaction survey
                        - Offer loyalty program enrollment
                        - Provide exceptional customer service
                        """)
                    else:  # Low risk (<30%)
                        st.write("""
                        **✅ Customer Likely to Stay:**
                        - Continue regular service
                        - Consider upselling opportunities
                        - Use as reference for testimonials
                        - Maintain current service quality
                        """)
                    
                    
                    # FEATURE IMPACT ANALYSIS
                    
                    # Show which features most influenced this specific prediction
                    if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
                        st.subheader("📊 Key Factors Influencing Prediction")
                        
                        # Extract feature importance (tree-based models) or coefficients (linear models)
                        if hasattr(model, 'feature_importances_'):
                            importances = model.feature_importances_
                        else:
                            importances = abs(model.coef_[0])  # Use absolute values of coefficients
                        
                        # Create DataFrame with feature impact data
                        feature_impact = pd.DataFrame({
                            'Feature': st.session_state.feature_columns,
                            'Impact': importances,
                            'Customer_Value': processed_input.iloc[0].values
                        }).sort_values('Impact', ascending=False).head(10)  # Top 10 most important
                        
                        # Create horizontal bar chart showing feature impact
                        fig_impact = px.bar(
                            feature_impact,
                            x='Impact',
                            y='Feature',
                            orientation='h',
                            title="Top 10 Factors Affecting This Prediction",
                            color='Impact',
                            color_continuous_scale='viridis'
                        )
                        st.plotly_chart(fig_impact, use_container_width=True)
                
                except Exception as e:
                    # Error handling for prediction failures
                    st.error(f"Error making prediction: {str(e)}")
                    st.write("Please check your input data and try again.")
        
        
        # CSV UPLOAD METHOD (BATCH PREDICTIONS)
        
        else:  # CSV Upload selected
            st.subheader("📁 Batch Prediction from CSV")
            
            # File uploader widget for CSV files
            uploaded_file = st.file_uploader("Upload CSV file for batch predictions", type=['csv'])
            
            if uploaded_file is not None:
                try:
                    # Read the uploaded CSV file
                    batch_data = pd.read_csv(uploaded_file)
                    
                    # Display preview of uploaded data
                    st.subheader("📊 Uploaded Data Preview")
                    st.dataframe(batch_data.head())  # Show first 5 rows
                    
                    # Button to trigger batch predictions
                    if st.button("🚀 Generate Batch Predictions", type="primary"):
                        with st.spinner("Making predictions..."):  # Show loading spinner
                            # Preprocess the entire batch dataset
                            processed_batch = preprocess_input(batch_data, st.session_state.data)
                            
                            # Get the selected trained model
                            model = st.session_state.models[selected_model]['model']
                            
                            # Make predictions for all rows
                            predictions = model.predict(processed_batch)
                            probabilities = model.predict_proba(processed_batch)[:, 1]  # Get churn probabilities
                            
                            # Add prediction results to original dataset
                            results_df = batch_data.copy()
                            results_df['Churn_Prediction'] = ['Yes' if p == 1 else 'No' for p in predictions]
                            results_df['Churn_Probability'] = probabilities
                            results_df['Risk_Level'] = [
                                'High' if p > 0.7 else 'Medium' if p > 0.3 else 'Low' 
                                for p in probabilities
                            ]
                            
                            # Display results table
                            st.subheader("📊 Prediction Results")
                            st.dataframe(results_df)
                            
                            
                            # BATCH PREDICTION SUMMARY STATISTICS
                            
                            col1, col2, col3 = st.columns(3)
                            
                            # Column 1: Total predicted churners
                            with col1:
                                churn_count = (predictions == 1).sum()
                                st.metric("Predicted Churners", churn_count)
                            
                            # Column 2: High risk customers count
                            with col2:
                                high_risk_count = (probabilities > 0.7).sum()
                                st.metric("High Risk Customers", high_risk_count)
                            
                            # Column 3: Average churn probability
                            with col3:
                                avg_probability = probabilities.mean()
                                st.metric("Average Churn Probability", f"{avg_probability:.1%}")
                            
                            
                            # CSV DOWNLOAD FUNCTIONALITY
                            
                            # Prepare CSV data for download
                            csv_buffer = io.StringIO()
                            results_df.to_csv(csv_buffer, index=False)
                            csv_string = csv_buffer.getvalue()
                            
                            # Create download button with timestamp in filename
                            st.download_button(
                                label="📥 Download Results as CSV",
                                data=csv_string,
                                file_name=f"churn_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                
                except Exception as e:
                    # Error handling for batch processing failures
                    st.error(f"Error processing batch predictions: {str(e)}")

# Page 6: Interpretation & Report
elif page == " Interpretation & Report":
    # Display main section header with custom HTML styling
    st.markdown('<h2 class="section-header"> Analysis Report & Interpretation</h2>', unsafe_allow_html=True)
    
    # Check if analysis data exists in session state
    if st.session_state.data is None:
        st.warning("⚠️ Please complete the analysis first.")
    else:
        # Get data from session state
        data = st.session_state.data
        
        
        # EXECUTIVE SUMMARY SECTION
        
        st.subheader("📊 Executive Summary")
        
        # Create two columns for summary layout
        col1, col2 = st.columns(2)
        
        # Initialize variables for churn metrics and best model
        churn_rate, churned_customers, best_model = None, None, None
        
        # Left column: Dataset overview metrics
        with col1:
            st.write("**Dataset Overview:**")
            # Display total number of customers (rows in dataset)
            st.write(f"• Total customers analyzed: {data.shape[0]:,}")
            # Display number of features (columns minus target variable)
            st.write(f"• Features considered: {data.shape[1] - 1}")
            
            # Calculate and display churn statistics if Churn column exists
            if 'Churn' in data.columns:
                churn_rate = (data['Churn'] == 'Yes').mean() * 100  # Percentage of churned customers
                churned_customers = (data['Churn'] == 'Yes').sum()  # Total number of churned customers
                st.write(f"• Overall churn rate: {churn_rate:.2f}%")
                st.write(f"• Customers who churned: {churned_customers:,}")
        
        # Right column: Best model performance metrics
        with col2:
            # Check if models exist in session state
            if st.session_state.models:
                st.write("**Model Performance:**")
                # Find the best model based on ROC-AUC score
                best_model = max(st.session_state.models.items(), key=lambda x: x[1]['roc_auc'])
                st.write(f"• Best performing model: {best_model[0]}")
                
                # Display all metrics for the best model
                for metric, value in best_model[1].items():
                    if isinstance(value, (int, float)):
                        # Format numeric values to 3 decimal places
                        st.write(f"• {metric.capitalize()}: {value:.3f}")
                    else:
                        # Display non-numeric values as-is
                        st.write(f"• {metric.capitalize()}: {value}")
        
        
        # FEATURE IMPORTANCE SECTION
        
        # Check if feature importance data exists and is not empty
        if "feature_importance" in st.session_state and not st.session_state.feature_importance.empty:
            st.subheader("🔍 Top Predictive Features")
            imp = st.session_state.feature_importance
            # Display bar chart of top 10 most important features
            st.bar_chart(imp.head(10))
        else:
            imp = None
        
        
        # INTERPRETATION SECTION
        
        st.subheader("📖 Interpretation & Insights")
        # Display stored interpretation text or default message
        st.write(st.session_state.get("interpretation_text", "No detailed interpretation available. Please ensure the model has been analyzed."))
        
       
        # CONCLUSION & RECOMMENDATIONS SECTION
        
        st.subheader("📝 Conclusion & Recommendations")
        
        # Generate dynamic conclusion if not already stored in session state
        if "conclusion_text" in st.session_state and st.session_state["conclusion_text"]:
            # Use existing conclusion text
            conclusion_text = st.session_state["conclusion_text"]
        else:
            # Generate conclusion based on best model results
            if best_model:
                model_name = best_model[0]
                metrics = best_model[1]
                # Extract individual metrics
                acc = metrics['accuracy']
                prec = metrics['precision']
                rec = metrics['recall']
                f1 = metrics['f1']
                roc_auc = metrics['roc_auc']
                
                # Create formatted conclusion text with metrics and recommendations
                conclusion_text = (
                    f"Based on the analysis, the **{model_name}** model achieved the best performance "
                    f"with a ROC-AUC score of {roc_auc:.3f}, accuracy of {acc:.3f}, precision of {prec:.3f}, "
                    f"recall of {rec:.3f}, and F1-score of {f1:.3f}.\n\n"
                    "This indicates that the model is effective for predicting customer churn in this dataset. "
                    "We recommend prioritizing retention strategies for customers flagged as high-risk by the model, "
                    "such as targeted offers, proactive customer support, and loyalty programs."
                )
            else:
                # Default message if no models are available
                conclusion_text = "No model results available to generate a conclusion."
        
        # Display the conclusion text
        st.write(conclusion_text)
        
       
        # PDF EXPORT FUNCTIONALITY
        
        # Import required libraries for PDF generation
        from fpdf import FPDF
        import matplotlib.pyplot as plt
        import tempfile
        import os
        
        # PDF download button
        if st.button("📄 Download PDF Report"):
            # Define custom PDF class extending FPDF
            class PDF(FPDF):
                def header(self):
                    """Add header to each PDF page"""
                    self.set_font('Helvetica', 'B', 16)
                    self.cell(0, 10, "Telco Churn Analysis Report", ln=True, align='C')
                    self.ln(5)
                
                def chapter_title(self, title):
                    """Format chapter titles"""
                    self.set_font('Helvetica', 'B', 14)
                    self.cell(0, 10, title, ln=True)
                    self.ln(3)
                
                def chapter_body(self, body):
                    """Format chapter body text"""
                    self.set_font('Helvetica', '', 12)
                    self.multi_cell(0, 8, body)
                    self.ln()
            
            # Create PDF instance and add first page
            pdf = PDF()
            pdf.add_page()
            
            # Add Executive Summary section to PDF
            pdf.chapter_title("Executive Summary")
            summary_text = (
                f"Total customers analyzed: {data.shape[0]:,}\n"
                f"Features considered: {data.shape[1] - 1}\n"
            )
            # Add churn statistics if available
            if churn_rate is not None:
                summary_text += f"Churn rate: {churn_rate:.2f}%\nCustomers who churned: {churned_customers:,}\n"
            pdf.chapter_body(summary_text)
            
            # Add Model Performance section to PDF
            if best_model:
                pdf.chapter_title("Model Performance")
                # Add each metric as a separate line
                for k, v in best_model[1].items():
                    if isinstance(v, (int, float)):
                        pdf.cell(0, 8, f"{k.capitalize()}: {v:.3f}", ln=True)
                    else:
                        pdf.cell(0, 8, f"{k.capitalize()}: {v}", ln=True)
                pdf.ln()
            
            # Add Top Features Chart to PDF (if feature importance data exists)
            if imp is not None:
                pdf.chapter_title("Top 10 Predictive Features")
                
                # Create matplotlib figure for feature importance
                fig, ax = plt.subplots()
                imp.head(10).plot(kind='bar', ax=ax, legend=False)
                ax.set_ylabel("Importance Score")
                ax.set_title("Top 10 Features")
                
                # Save chart as temporary image file
                temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                plt.tight_layout()
                plt.savefig(temp_img.name)
                plt.close(fig)
                
                # Add image to PDF and clean up temp file
                pdf.image(temp_img.name, w=170)
                os.unlink(temp_img.name)  # Delete temporary file
                pdf.ln()
            
            # Add Interpretation section to PDF
            pdf.chapter_title("Interpretation & Insights")
            pdf.chapter_body(st.session_state.get("interpretation_text", ""))
            
            # Add Conclusion section to PDF
            pdf.chapter_title("Conclusion & Recommendations")
            pdf.chapter_body(conclusion_text)
            
            # Generate PDF output and create download button
            pdf_output = pdf.output(dest='S').encode('latin1')
            st.download_button(
                label="📥 Download PDF",
                data=pdf_output,
                file_name="churn_analysis_report.pdf",
                mime="application/pdf"
            )





