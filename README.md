# Automated Underwriting Platform

A comprehensive property risk assessment platform that combines document analysis, computer vision, and statistical analysis to provide detailed risk assessments for insurance underwriting.

## 🚀 Features

### Core Capabilities

- **Document Analysis**: Extract and analyze property information from PDFs, DOCX, and text files
- **Computer Vision**: Analyze property images for condition assessment and hazard detection
- **Enhanced Risk Assessment**: Statistical analysis with confidence intervals, outlier detection, and correlation analysis
- **Claims Classification**: Automatically categorize claims by type (Medical, Property, Auto, Liability, etc.)
- **Data Quality Assessment**: Grade data quality and provide confidence scores

### Advanced Analytics

- **Statistical Analysis**: Z-scores, percentiles, and confidence intervals
- **Outlier Detection**: Identify unusual values in property data
- **Correlation Analysis**: Find relationships between risk factors
- **Risk Components**: Detailed breakdown of property condition, hazards, age, location, and maintenance risks

## 📋 Requirements

- Python 3.8+
- Streamlit
- See `requirements.txt` for full dependencies

## 🛠️ Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd Automated-Underwriting-Platform
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**

   ```bash
   streamlit run app.py
   ```

4. **Access the app**
   - Open your browser to `http://localhost:8501`
   - The app will automatically open in your default browser

## 📖 Usage

### Basic Usage (No API Key Required)

1. Upload property documents (PDF, DOCX, TXT) and images (PNG, JPG)
2. Enable the analysis options you want:
   - ✅ Document Analysis
   - ✅ Computer Vision Analysis
   - ✅ Enhanced Risk Assessment
3. Click "Process Documents" to analyze your files
4. Review the detailed risk assessment results

### Enhanced Analysis (With Grok API Key)

1. Get a Grok API key from [Groq](https://console.groq.com/)
2. Enter your API key in the sidebar
3. Upload and process documents as above
4. Get AI-enhanced analysis with additional insights

## 📊 Understanding Results

### Risk Assessment

- **Risk Score**: 0-1 scale (higher = more risk)
- **Risk Category**: Low/Medium/High/Very High Risk
- **Confidence Level**: How confident the assessment is
- **Processing Priority**: Recommended action level

### Detailed Analysis

Click "📊 Detailed Risk Analysis" to see:

- **Risk Components**: Breakdown by property condition, hazards, age, etc.
- **Statistical Analysis**: Z-scores and percentiles for property values
- **Outlier Detection**: Unusual values flagged with severity levels
- **Correlation Analysis**: Relationships between different factors
- **Data Quality**: Overall grade and individual metrics

### Claims Classification

- **Claim Type**: Medical, Property, Auto, Liability, Workers Comp, Life
- **Classification Confidence**: How certain the classification is
- **Other Indicators**: Additional claim types detected

## 🔧 Configuration

### Risk Thresholds

The system uses configurable risk thresholds:

- **Low Risk**: ≤ 0.3
- **Medium Risk**: 0.3 - 0.6
- **High Risk**: 0.6 - 0.8
- **Very High Risk**: > 0.8

### Statistical Parameters

- **Confidence Level**: 95% (configurable)
- **Outlier Threshold**: 2.0 standard deviations
- **Correlation Threshold**: 0.3 minimum correlation

## 📁 Project Structure

```
Automated-Underwriting-Platform/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .gitignore                      # Git ignore rules
├── yolov8n.pt                      # YOLO model for object detection
└── src/
    ├── __init__.py
    ├── document_analyzer.py        # Document processing and text extraction
    ├── vision_analyzer.py          # Computer vision and image analysis
    ├── risk_assessor.py            # Basic risk assessment engine
    └── enhanced_risk_assessor.py   # Advanced statistical risk assessment
```

## 🚨 Troubleshooting

### Common Issues

1. **CUDA/MPS Warning**: Normal - the app works on CPU, GPU just makes it faster
2. **API Key Issues**: Check your Groq API key is valid and has credits
3. **File Upload Errors**: Ensure files are under 200MB and in supported formats
4. **Analysis Errors**: Try processing files individually to isolate issues

### Performance Tips

- Use GPU if available for faster image processing
- Process smaller batches of files for better performance
- Close other applications to free up memory

## 🔒 Privacy & Security

- All processing is done locally (except Grok API calls)
- No data is stored permanently
- API keys are only used for the current session
- Uploaded files are processed in memory only

## 📈 Performance Metrics

The app tracks:

- Files processed per session
- Processing time
- Success rates
- Analysis completion rates

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is for educational and demonstration purposes.

---

**Ready to get started?** Run `streamlit run app.py` and upload your first document!
