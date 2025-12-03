# 🏠 House Price Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)](https://scikit-learn.org/)

An intelligent **machine learning application** that analyzes housing data and provides real-time house price predictions through an intuitive web interface. This project demonstrates end-to-end data science workflow from exploratory data analysis to model deployment.

## 🚀 Features

- **📊 Interactive Data Analysis**: Comprehensive Jupyter notebook with data exploration and visualization
- **🤖 Multiple ML Models**: Linear Regression and Random Forest algorithms for price prediction
- **🎨 Beautiful Web Interface**: Modern Streamlit dashboard with customizable backgrounds
- **⚡ Real-time Predictions**: Instant house price estimates based on user inputs
- **📈 Model Performance Metrics**: R² scores and accuracy measurements
- **🎯 User-friendly Design**: Intuitive sliders and input controls

## 📁 Project Structure

```
House-Price-Predictor/
├── 📓 1.Data Analysis.ipynb    # Exploratory data analysis & model experiments
├── 📊 housing.csv              # Training dataset (California housing data)
├── 🌐 streamlit_app.py         # Interactive web application
├── 📋 requirements.txt         # Python dependencies
├── 📜 LICENSE                  # MIT License
└── 📖 README.md               # Project documentation
```

## 🛠️ Technology Stack

- **Backend**: Python 3.8+
- **ML Libraries**: Scikit-learn, Pandas, NumPy
- **Frontend**: Streamlit
- **Data Analysis**: Jupyter Notebook
- **Visualization**: Matplotlib, Seaborn (in notebook)

## ⚡ Quick Start

### Prerequisites

- Python 3.8 or higher
- Git (optional)
- PowerShell (Windows) or Terminal (Mac/Linux)

### Installation & Setup

1. **📥 Clone the repository**
   ```powershell
   git clone https://github.com/sandudul/House-Price-Predictor.git
   cd House-Price-Predictor
   ```

2. **🐍 Create virtual environment** (Recommended)
   ```powershell
   python -m venv .\.venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **📦 Install dependencies**
   ```powershell
   pip install -r .\requirements.txt
   ```

4. **🚀 Launch the application**
   ```powershell
   streamlit run .\streamlit_app.py
   ```

5. **🌐 Open in browser**
   
   Navigate to `http://localhost:8501` to access the application

## 🎯 How to Use

1. **Launch the App**: Follow the installation steps above
2. **Input House Features**: Use the sidebar sliders to adjust:
   - 🏠 House age
   - 👥 Average rooms
   - 🏘️ Population density
   - 💰 Median income
   - 📍 Geographic coordinates
3. **Get Predictions**: View real-time price estimates from multiple models
4. **Compare Models**: Analyze R² scores to understand model performance

## 📊 Dataset Information

The application uses the **California Housing Dataset**, which includes:

| Feature | Description |
|---------|-------------|
| `housing_median_age` | Median age of houses in the block |
| `total_rooms` | Total number of rooms in the block |
| `total_bedrooms` | Total number of bedrooms in the block |
| `population` | Population in the block |
| `households` | Number of households in the block |
| `median_income` | Median income of households |
| `latitude` | Latitude coordinate |
| `longitude` | Longitude coordinate |

**Target Variable**: `median_house_value` (in hundreds of thousands of dollars)

## 🤖 Models Used

| Model | Description | Use Case |
|-------|-------------|----------|
| **Linear Regression** | Simple linear relationship modeling | Baseline performance & interpretability |
| **Random Forest** | Ensemble method with decision trees | Handling non-linear patterns & feature interactions |

## 🔧 Advanced Configuration

### Custom Background Images
Place your image file in the project directory and modify the background path in `streamlit_app.py`:
```python
set_background("your_image.jpg")
```

### Model Optimization
For production deployment, consider:
- Saving trained models using `joblib` or `pickle`
- Implementing model versioning
- Adding cross-validation
- Feature engineering enhancements

## 📈 Performance Metrics

The application displays model performance using:
- **R² Score**: Coefficient of determination
- **Prediction Accuracy**: Real-time validation
- **Training Time**: Model efficiency metrics

## 🚨 Troubleshooting

| Issue | Solution |
|-------|----------|
| **Streamlit won't start** | Ensure virtual environment is activated and dependencies are installed |
| **Missing dataset** | Verify `housing.csv` exists in the project root directory |
| **Import errors** | Run `pip install -r requirements.txt` to install missing packages |
| **Performance issues** | Consider using saved models instead of training at startup |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎓 Learning Objectives

This project demonstrates:
- **Data Science Workflow**: From EDA to model deployment
- **Machine Learning**: Regression algorithms and performance evaluation
- **Web Development**: Interactive dashboards with Streamlit
- **Best Practices**: Code organization, documentation, and version control

## 📧 Contact

**Sandu** - [@sandudul](https://github.com/sandudul)

Project Link: [https://github.com/sandudul/House-Price-Predictor](https://github.com/sandudul/House-Price-Predictor)

---

⭐ **Star this repository** if you found it helpful!
