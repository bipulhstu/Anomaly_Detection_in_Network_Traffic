"""
Streamlit App for Network Traffic Anomaly Detection
Uses Isolation Forest model to detect anomalies in network traffic data
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os

# Page configuration
st.set_page_config(
    page_title="Network Traffic Anomaly Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 1rem;
    }
    h2 {
        color: #2ca02c;
        padding-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    """Load the trained model and scaler"""
    try:
        model_path = 'isolation_forest_model.joblib'
        scaler_path = 'scaler.joblib'
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            st.error("⚠️ Model files not found! Please ensure 'isolation_forest_model.joblib' and 'scaler.joblib' are in the app directory.")
            return None, None
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None

# Feature engineering function
def create_features(df, scaler):
    """Scale the network traffic data for the model"""
    df = df.copy()
    
    # Scale data - this is the ONLY feature the model expects
    df['network_in_scaled'] = scaler.transform(df[['network_in']])
    
    return df

# Prediction function
def predict_anomalies(df, model, scaler):
    """Predict anomalies in the dataframe"""
    df_processed = create_features(df, scaler)
    
    if len(df_processed) == 0:
        st.warning("⚠️ Not enough data points for prediction.")
        return None
    
    # Get features - the model was trained on ONLY network_in_scaled
    X = df_processed[['network_in_scaled']]
    
    # Predict
    df_processed['anomaly_score'] = model.decision_function(X)
    df_processed['anomaly'] = model.predict(X)
    df_processed['prediction'] = df_processed['anomaly'].apply(
        lambda x: 'Normal' if x == 1 else 'Anomaly'
    )
    
    return df_processed

# Visualization functions
def plot_time_series(df, title="Network Traffic Over Time"):
    """Plot time series with anomalies highlighted"""
    fig = go.Figure()
    
    # Plot normal traffic
    normal_data = df[df['anomaly'] == 1]
    fig.add_trace(go.Scatter(
        x=normal_data.index,
        y=normal_data['network_in'],
        mode='lines',
        name='Normal Traffic',
        line=dict(color='#2ca02c', width=2)
    ))
    
    # Plot anomalies
    anomaly_data = df[df['anomaly'] == -1]
    fig.add_trace(go.Scatter(
        x=anomaly_data.index,
        y=anomaly_data['network_in'],
        mode='markers',
        name='Anomalies',
        marker=dict(color='red', size=10, symbol='x')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Time',
        yaxis_title='Network Traffic (Bytes)',
        hovermode='x unified',
        height=500
    )
    
    return fig

def plot_anomaly_scores(df):
    """Plot anomaly scores distribution"""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=df['anomaly_score'],
        nbinsx=50,
        name='Anomaly Score Distribution',
        marker_color='#1f77b4'
    ))
    
    fig.update_layout(
        title='Distribution of Anomaly Scores',
        xaxis_title='Anomaly Score',
        yaxis_title='Frequency',
        height=400
    )
    
    return fig

# Main app
def main():
    # Load model
    global model, scaler
    model, scaler = load_model_and_scaler()
    
    if model is None or scaler is None:
        st.stop()
    
    # Header
    st.title("🛡️ Network Traffic Anomaly Detection System")
    st.markdown("""
    This application uses **Isolation Forest** machine learning algorithm to detect anomalies in network traffic data.
    Upload your network traffic data or use the sample data to identify potential security threats or unusual activity.
    """)
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Input method selection
    input_method = st.sidebar.radio(
        "Choose Input Method:",
        ["📤 Upload CSV File", "✏️ Manual Input", "📊 Use Sample Data"],
        index=2  # Default to "Use Sample Data"
    )
    
    df = None
    
    # Input handling
    if input_method == "📤 Upload CSV File":
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Upload Instructions")
        st.sidebar.info("""
        Your CSV file should have:
        - **timestamp** column (datetime)
        - **value** column (network traffic in bytes)
        """)
        
        uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Rename columns
                if 'value' in df.columns and 'timestamp' in df.columns:
                    df.rename(columns={'value': 'network_in', 'timestamp': 'Timestamp'}, inplace=True)
                elif 'network_in' not in df.columns or 'Timestamp' not in df.columns:
                    st.error("❌ CSV must contain 'timestamp' and 'value' columns (or 'Timestamp' and 'network_in')")
                    st.stop()
                
                # Convert timestamp
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                df.set_index('Timestamp', inplace=True)
                
                # Handle missing values
                df = df.fillna(df['network_in'].mean())
                
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
                st.stop()
    
    elif input_method == "✏️ Manual Input":
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Manual Input")
        
        num_points = st.sidebar.number_input(
            "Number of data points:",
            min_value=5,
            max_value=100,
            value=20,
            step=1
        )
        
        network_values = st.sidebar.text_area(
            "Enter network traffic values (comma-separated):",
            value=",".join([str(np.random.randint(1000, 10000)) for _ in range(num_points)]),
            height=150
        )
        
        if st.sidebar.button("🚀 Analyze Data"):
            try:
                values = [float(x.strip()) for x in network_values.split(',')]
                
                # Create timestamps
                timestamps = [datetime.now() - timedelta(minutes=5*i) for i in range(len(values))]
                timestamps.reverse()
                
                df = pd.DataFrame({
                    'Timestamp': timestamps,
                    'network_in': values
                })
                df.set_index('Timestamp', inplace=True)
                
            except Exception as e:
                st.error(f"❌ Error parsing input: {str(e)}")
                st.stop()
    
    else:  # Use Sample Data
        st.sidebar.markdown("---")
        st.sidebar.info("Loading sample data from 'ec2_network_in_257a54.csv'")
        
        try:
            df = pd.read_csv('ec2_network_in_257a54.csv', nrows=1000)
            df.rename(columns={'value': 'network_in', 'timestamp': 'Timestamp'}, inplace=True)
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df.set_index('Timestamp', inplace=True)
            df = df.fillna(df['network_in'].mean())
        except FileNotFoundError:
            st.error("❌ Sample data file 'ec2_network_in_257a54.csv' not found!")
            st.stop()
        except Exception as e:
            st.error(f"❌ Error loading sample data: {str(e)}")
            st.stop()
    
    # Process data if available
    if df is not None:
        st.success(f"✅ Successfully loaded {len(df)} data points!")
        
        # Show raw data
        with st.expander("📋 View Raw Data", expanded=False):
            st.dataframe(df.head(20), use_container_width=True)
        
        # Run prediction
        with st.spinner("🔍 Analyzing network traffic..."):
            results = predict_anomalies(df, model, scaler)
        
        if results is not None:
            # Calculate statistics
            total_points = len(results)
            anomalies_count = len(results[results['anomaly'] == -1])
            normal_count = len(results[results['anomaly'] == 1])
            anomaly_percentage = (anomalies_count / total_points) * 100
            
            # Display metrics
            st.markdown("## 📊 Analysis Results")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="Total Data Points",
                    value=f"{total_points}",
                    delta=None
                )
            
            with col2:
                st.metric(
                    label="Normal Traffic",
                    value=f"{normal_count}",
                    delta=f"{100-anomaly_percentage:.1f}%"
                )
            
            with col3:
                st.metric(
                    label="Anomalies Detected",
                    value=f"{anomalies_count}",
                    delta=f"{anomaly_percentage:.1f}%",
                    delta_color="inverse"
                )
            
            with col4:
                avg_score = results['anomaly_score'].mean()
                st.metric(
                    label="Avg Anomaly Score",
                    value=f"{avg_score:.3f}",
                    delta=None
                )
            
            # Visualization
            st.markdown("## 📈 Visualizations")
            
            # Time series plot
            fig1 = plot_time_series(results)
            st.plotly_chart(fig1, use_container_width=True)
            
            # Distribution plots
            col1, col2 = st.columns(2)
            
            with col1:
                fig2 = px.histogram(
                    results,
                    x='network_in',
                    color='prediction',
                    title='Network Traffic Distribution',
                    labels={'network_in': 'Network Traffic (Bytes)', 'count': 'Frequency'},
                    color_discrete_map={'Normal': '#2ca02c', 'Anomaly': 'red'}
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            with col2:
                fig3 = plot_anomaly_scores(results)
                st.plotly_chart(fig3, use_container_width=True)
            
            # Anomaly details
            if anomalies_count > 0:
                st.markdown("## 🚨 Detected Anomalies")
                
                anomaly_df = results[results['anomaly'] == -1][['network_in', 'anomaly_score', 'prediction']].copy()
                anomaly_df['network_in'] = anomaly_df['network_in'].apply(lambda x: f"{x:,.0f} bytes")
                anomaly_df['anomaly_score'] = anomaly_df['anomaly_score'].apply(lambda x: f"{x:.4f}")
                
                st.dataframe(
                    anomaly_df,
                    use_container_width=True,
                    column_config={
                        "network_in": "Network Traffic",
                        "anomaly_score": "Anomaly Score",
                        "prediction": "Status"
                    }
                )
                
                # Download results
                csv = results.to_csv()
                st.download_button(
                    label="📥 Download Full Results as CSV",
                    data=csv,
                    file_name=f'anomaly_detection_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                    mime='text/csv'
                )
            else:
                st.success("✅ No anomalies detected! All traffic appears normal.")
            
            # Interpretation guide
            with st.expander("ℹ️ How to Interpret Results", expanded=False):
                st.markdown("""
                ### Understanding Anomaly Scores
                
                - **Anomaly Score**: A measure of how different a data point is from normal patterns
                    - Higher scores indicate more normal behavior
                    - Lower (negative) scores indicate potential anomalies
                
                - **Prediction**:
                    - **Normal**: Traffic pattern matches expected behavior
                    - **Anomaly**: Traffic pattern deviates significantly from the norm
                
                ### What Anomalies Might Indicate
                
                - 🔴 **Security Threats**: DDoS attacks, data exfiltration attempts
                - ⚠️ **System Issues**: Hardware failures, network congestion
                - 🔧 **Configuration Changes**: New services, policy updates
                - 📊 **Unusual Activity**: Legitimate but uncommon traffic patterns
                
                ### Recommended Actions
                
                1. Investigate the time and context of anomalies
                2. Check system logs for corresponding events
                3. Verify with network administrators
                4. Update monitoring thresholds if needed
                """)

if __name__ == '__main__':
    main()

