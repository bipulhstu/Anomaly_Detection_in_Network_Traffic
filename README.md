# 🛡️ Network Traffic Anomaly Detection

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 🚀 [**Live Demo**](https://anomaly-d.streamlit.app/) 

An intelligent anomaly detection system for network traffic using machine learning techniques. This project employs **Isolation Forest** algorithm to identify unusual patterns in network traffic data that may indicate security threats, system failures, or other anomalous behavior.

**Try it now:** [https://anomaly-d.streamlit.app/](https://anomaly-d.streamlit.app/)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Deployment](#deployment)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

Network traffic anomaly detection is crucial for:
- **Cybersecurity**: Identifying potential DDoS attacks, intrusions, or data breaches
- **Performance Monitoring**: Detecting unusual traffic patterns that may indicate system issues
- **Capacity Planning**: Understanding traffic trends and patterns
- **Compliance**: Ensuring network behavior aligns with security policies

This project uses unsupervised machine learning (Isolation Forest) to automatically learn normal traffic patterns and flag deviations without requiring labeled training data.

## ✨ Features

- **Unsupervised Learning**: No need for labeled anomaly data
- **Real-time Detection**: Fast inference for live traffic monitoring
- **Interactive Dashboard**: Streamlit-based web application for easy visualization
- **Feature Engineering**: Automated creation of temporal features (lagged values)
- **Model Persistence**: Save and load trained models for production use
- **Comprehensive Visualization**: Time series plots, distributions, and anomaly scores
- **Multiple Input Methods**: Upload CSV, manual input, or use sample data
- **Exportable Results**: Download detection results as CSV

## 📊 Dataset

The project uses the **EC2 Network Traffic dataset** from the Numenta Anomaly Benchmark (NAB):

- **Source**: [Kaggle - NAB Dataset](https://www.kaggle.com/datasets/boltzmannbrain/nab/data?select=realKnownCause)
- **File**: `ec2_network_in_257a54.csv`
- **Features**:
  - `timestamp`: Date and time of measurement
  - `value`: Network traffic volume (bytes)
- **Size**: ~4,000+ data points collected at 5-minute intervals

### Data Characteristics

- Time series data with temporal dependencies
- Contains real anomalies from known causes
- Exhibits both trend and seasonal patterns
- Includes various types of anomalous behavior (spikes, drops, sustained changes)

## 🗂️ Project Structure

```
Anomaly_Detection_in_Network_Traffic/
│
├── 📊 Anomaly_Detection_in_Network_Traffic.ipynb  # Complete analysis & EDA notebook
├── 🏋️ Anomaly_Detection_Training.ipynb            # Simplified model training
├── 🌐 app.py                                      # Streamlit web application
├── 🚀 deployment.py                               # Flask API deployment
├── 📈 ec2_network_in_257a54.csv                  # NAB dataset (4,033 records)
├── 🤖 isolation_forest_model.joblib              # Trained Isolation Forest model
├── ⚖️ scaler.joblib                              # Fitted StandardScaler
├── 📦 requirements.txt                           # Python dependencies
└── 📖 README.md                                  # Project documentation
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/Anomaly_Detection_in_Network_Traffic.git
cd Anomaly_Detection_in_Network_Traffic
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Required Libraries

```txt
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
streamlit>=1.28.0
plotly>=5.17.0
joblib>=1.3.0
statsmodels>=0.14.0
flask>=2.3.0
flask-cors>=4.0.0
```

## 💻 Usage

### 1. Training the Model

Run the Jupyter notebook to train the model:

```bash
jupyter notebook Anomaly_Detection_in_Network_Traffic.ipynb
```

Or train from scratch:

```python
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

# Load and preprocess data
df = pd.read_csv('ec2_network_in_257a54.csv')
# ... (preprocessing steps)

# Train model
model = IsolationForest(n_estimators=50, contamination='auto', random_state=42)
model.fit(X_train)

# Save model
joblib.dump(model, 'isolation_forest_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
```

### 2. Running the Streamlit App

#### Option A: Use the Live Demo (Recommended)
🌐 **[Try the live app here: https://anomaly-d.streamlit.app/](https://anomaly-d.streamlit.app/)**

#### Option B: Run Locally
Launch the interactive web application:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

**App Features**:
- 📤 Upload your own CSV files
- ✏️ Enter data manually  
- 📊 Use sample data for testing (default)
- 📈 Interactive visualizations with Plotly
- 📊 Real-time anomaly detection results
- 📥 Download results as CSV
- ℹ️ Comprehensive interpretation guide

### 3. Using the Flask API

For production deployment or integration with other systems:

```bash
python deployment.py
```

**API Endpoints**:

- `GET /` - API status check
- `POST /predict` - Detect anomalies

**Example Request**:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"network_in": 12345}'
```

**Example Response**:
```json
{
  "network_in": 12345,
  "prediction": -1,
  "result": "Anomaly",
  "description": "The network traffic is classified as Anomaly."
}
```

## 🤖 Model Details

### Algorithm: Isolation Forest

**Why Isolation Forest?**
- Specifically designed for anomaly detection
- Works well with high-dimensional data
- Fast training and prediction
- No need for labeled data
- Effective at isolating anomalies that differ from the majority

**How it works**:
1. Randomly selects a feature
2. Randomly selects a split value between min and max
3. Recursively partitions data into trees
4. Anomalies are isolated faster (shorter paths in trees)
5. Anomaly score based on average path length across all trees

### Feature Engineering

The model uses several engineered features:

1. **Scaled Network Traffic**: Standardized using StandardScaler
2. **Lagged Features**: Past 4 time steps (network_in_lag_1 to network_in_lag_4)
   - Captures temporal dependencies
   - Helps identify sequential anomalies

### Hyperparameters

After grid search optimization:

- `n_estimators`: 50 (number of trees)
- `contamination`: 'auto' (proportion of outliers)
- `random_state`: 42 (reproducibility)

### Model Performance

- **Training Data**: 80% of dataset (temporal split to preserve time series structure)
- **Test Data**: 20% of dataset
- **Model Type**: Unsupervised learning (no labeled anomalies required)
- **Evaluation**: Visual inspection, anomaly score distribution, and F1-score optimization
- **Hyperparameter Tuning**: GridSearchCV with custom F1-scorer for anomaly detection

## 🌐 Deployment

### Live Deployment

🌐 **Production App**: [https://anomaly-d.streamlit.app/](https://anomaly-d.streamlit.app/)

The application is deployed on **Streamlit Cloud** and includes:
- Real-time anomaly detection
- Interactive visualizations
- Sample data pre-loaded
- CSV upload functionality
- Results download capability

### Local Deployment

1. **Streamlit** (Interactive Dashboard):
```bash
streamlit run app.py
```

2. **Flask API** (REST API):
```bash
python deployment.py
```

### Cloud Deployment Options

#### Option 1: Streamlit Cloud (Current Deployment)

✅ **Already deployed at**: [https://anomaly-d.streamlit.app/](https://anomaly-d.streamlit.app/)

To deploy your own:
1. Push code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect repository and deploy

#### Option 2: Docker

Create `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t anomaly-detection .
docker run -p 8501:8501 anomaly-detection
```

#### Option 3: Cloud Platforms

- **AWS**: Deploy on EC2, ECS, or Lambda
- **Google Cloud**: Use Cloud Run or App Engine
- **Azure**: Deploy on App Service or Container Instances
- **Heroku**: Simple deployment with Procfile

## 📈 Results

### Key Findings

- ✅ Successfully detected anomalous traffic patterns in EC2 network data
- ✅ Low false positive rate with 'auto' contamination setting
- ✅ Model generalizes well to unseen data with temporal split
- ✅ Fast inference time suitable for real-time monitoring
- ✅ Comprehensive EDA revealed seasonal patterns and trends
- ✅ Hyperparameter tuning improved model performance

### Visualizations

The project includes comprehensive visualizations:

1. **📈 Time Series Plot**: Network traffic over time with anomalies highlighted in red
2. **📊 Distribution Analysis**: Histogram showing normal vs anomalous traffic patterns  
3. **🎯 Anomaly Scores**: Distribution of decision function scores
4. **🔄 Time Series Decomposition**: Trend, seasonal, and residual components
5. **📈 Autocorrelation Plot**: Temporal dependencies in traffic patterns
6. **📊 Rolling Statistics**: Moving averages and standard deviations
7. **📦 Boxplot Analysis**: Outlier identification and quartile analysis

### Live Demo Results

🌐 **Try it yourself**: [https://anomaly-d.streamlit.app/](https://anomaly-d.streamlit.app/)

**Example Output from Sample Data**:
```
📊 Analysis Results:
├── Total Data Points: 1,000
├── Normal Traffic: 982 (98.2%)
├── Anomalies Detected: 18 (1.8%)
└── Average Anomaly Score: 0.142

🚨 Detected Anomalies:
├── High traffic spikes (>15,000 bytes)
├── Unusual low traffic periods (<500 bytes)  
└── Sudden traffic pattern changes
```

## 🔮 Future Improvements

### Model Enhancements

- [ ] Implement ensemble methods (combine multiple algorithms)
- [ ] Add LSTM/GRU autoencoders for deep learning approach
- [ ] Incorporate multi-variate features (e.g., network out, CPU usage)
- [ ] Implement online learning for model updates

### Feature Engineering

- [ ] Add statistical features (rolling mean, std, percentiles)
- [ ] Include time-based features (hour, day of week, is_weekend)
- [ ] Calculate rate of change and acceleration
- [ ] Add frequency domain features (FFT)

### Deployment

- [ ] Real-time streaming data processing
- [ ] Integration with monitoring systems (Grafana, Prometheus)
- [ ] Automated alerting system
- [ ] A/B testing framework for model comparison

### User Experience

- [ ] Add user authentication
- [ ] Implement feedback mechanism for false positives
- [ ] Create customizable alert thresholds
- [ ] Add historical trend analysis

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Guidelines

- Follow PEP 8 style guide
- Add docstrings to functions
- Include unit tests for new features
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Bipul**

- 🌐 **Live Demo**: [https://anomaly-d.streamlit.app/](https://anomaly-d.streamlit.app/)
- 💻 **GitHub**: [@bipul](https://github.com/bipul)
- 📧 **Contact**: For questions or collaboration opportunities

## 🙏 Acknowledgments

- **Numenta Anomaly Benchmark (NAB)** for providing the dataset
- **Kaggle** for hosting the data
- **Scikit-learn** team for the excellent ML library
- **Streamlit** for the amazing web framework

## 📚 References

1. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest. *ICDM '08: Proceedings of the 2008 Eighth IEEE International Conference on Data Mining*.
2. Ahmad, S., Lavin, A., Purdy, S., & Agha, Z. (2017). Unsupervised real-time anomaly detection for streaming data. *Neurocomputing*.
3. Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly detection: A survey. *ACM Computing Surveys*.

## 📞 Support

For questions or issues:
- 🌐 **Try the Live Demo**: [https://anomaly-d.streamlit.app/](https://anomaly-d.streamlit.app/)
- 💻 **Open an issue on GitHub**
- 📧 **Contact the author for collaboration**

---

**⭐ If you find this project helpful, please consider giving it a star!**

**🚀 [Experience the live demo here!](https://anomaly-d.streamlit.app/)**

