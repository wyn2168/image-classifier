import streamlit as st
import requests
import numpy as np
from PIL import Image
import io
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# Configure the page
st.set_page_config(
    page_title="🤖 AI Image Classifier",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #4CAF50, #45a049);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .upload-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
    }
    .result-section {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    .error-section {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .stProgress .st-bo {
        background-color: #00ff88;
    }
</style>
""", unsafe_allow_html=True)

# Configuration - CHANGE THIS TO YOUR LOCAL MACHINE'S IP
# Find your IP with: ipconfig (Windows) or ifconfig (Mac/Linux)
API_BASE_URL = os.getenv("API_BASE_URL", " https://7df135f5df69.ngrok-free.app")

# For development, you can also use ngrok to expose your local server
# Download ngrok from https://ngrok.com/ and run: ngrok http 8000
# Then use the ngrok URL like: https://abc123.ngrok.io

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'current_prediction' not in st.session_state:
    st.session_state.current_prediction = None

def get_local_ip_instructions():
    """Show instructions for setting up local IP"""
    with st.expander("🔧 Setup Instructions", expanded=False):
        st.markdown("""
        **To connect to your local GPU machine:**
        
        1. **Find your local IP address:**
           - Windows: Open CMD and run `ipconfig`
           - Mac/Linux: Open Terminal and run `ifconfig`
           - Look for your local IP (usually 192.168.x.x or 10.x.x.x)
        
        2. **Update the API_BASE_URL:**
           - Replace `YOUR_LOCAL_IP` with your actual IP address
           - Example: `http://192.168.1.100:8000`
        
        3. **Alternative - Use ngrok (easier):**
           - Download from https://ngrok.com/
           - Run: `ngrok http 8000`
           - Copy the https URL and set it as API_BASE_URL
        
        4. **Make sure your FastAPI server is running:**
           - Run `python main.py` on your local machine
           - Your GPU will be used automatically if available
        """)

def test_backend_connection():
    """Test connection to the backend"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return True, data
        else:
            return False, {"error": f"Status code: {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return False, {"error": str(e)}

def get_model_info():
    """Get detailed model information"""
    try:
        response = requests.get(f"{API_BASE_URL}/model-info", timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def predict_image(uploaded_file):
    """Make prediction on uploaded image"""
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        
        with st.spinner("🔮 AI is analyzing your image..."):
            response = requests.post(
                f"{API_BASE_URL}/predict",
                files=files,
                timeout=60  # Increased timeout for GPU processing
            )
        
        if response.status_code == 200:
            result = response.json()
            
            # Add to history
            st.session_state.prediction_history.append({
                "timestamp": datetime.now(),
                "filename": uploaded_file.name,
                "prediction": result["prediction"]["top_class"],
                "confidence": result["prediction"]["confidence"]
            })
            
            return True, result
        else:
            return False, f"Error {response.status_code}: {response.text}"
    
    except requests.exceptions.Timeout:
        return False, "⏰ Request timed out. The model might be processing a large image."
    except requests.exceptions.ConnectionError:
        return False, "🔌 Cannot connect to the backend. Make sure your FastAPI server is running."
    except Exception as e:
        return False, f"❌ Unexpected error: {str(e)}"

def display_prediction_results(result):
    """Display prediction results with beautiful formatting"""
    prediction = result["prediction"]
    
    # Main prediction
    st.markdown("### 🎯 Main Prediction")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"**Class:** `{prediction['top_class']}`")
        st.markdown(f"**Confidence:** `{prediction['confidence']:.1%}`")
    
    with col2:
        # Confidence gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prediction['confidence'] * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Confidence %"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
    
    # Top 5 predictions
    st.markdown("### 📊 Top 5 Predictions")
    
    top5_data = []
    for pred in prediction["top_5_predictions"]:
        top5_data.append({
            "Rank": pred["rank"],
            "Class": pred["class"],
            "Confidence": f"{pred['confidence']:.1%}",
            "Score": pred["confidence"]
        })
    
    df = pd.DataFrame(top5_data)
    
    # Create horizontal bar chart
    fig = px.bar(
        df, 
        x="Score", 
        y="Class", 
        orientation='h',
        color="Score",
        color_continuous_scale="viridis",
        title="Confidence Scores for Top 5 Predictions"
    )
    fig.update_layout(height=300, showlegend=False)
    fig.update_traces(texttemplate='%{x:.1%}', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed table
    st.dataframe(df[["Rank", "Class", "Confidence"]], use_container_width=True)
    
    # Model information
    with st.expander("🤖 Model Details", expanded=False):
        model_info = result.get("model_info", {})
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Model", model_info.get("model", "N/A"))
            st.metric("Device", model_info.get("device", "N/A"))
        
        with col2:
            st.metric("Total Classes", model_info.get("total_classes", "N/A"))
            st.metric("Image Size", result.get("image_size", "N/A"))

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 AI Image Classifier</h1>', unsafe_allow_html=True)
    st.markdown("**Upload an image and let AI identify what's in it!**")
    
    # Setup instructions
    get_local_ip_instructions()
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 System Status")
        
        # Test backend connection
        if st.button("🔍 Check Connection", use_container_width=True):
            is_connected, data = test_backend_connection()
            
            if is_connected:
                st.success("✅ Connected to Backend!")
                
                # Display system info
                with st.expander("💻 System Info", expanded=True):
                    st.json(data)
                    
                # Get detailed model info
                model_info = get_model_info()
                if model_info:
                    with st.expander("🤖 Model Info", expanded=False):
                        st.json(model_info)
            else:
                st.error("❌ Cannot connect to backend")
                st.error(data.get("error", "Unknown error"))
                st.info("Make sure your FastAPI server is running on your local machine!")
        
        # Prediction history
        if st.session_state.prediction_history:
            st.header("📈 Recent Predictions")
            for i, pred in enumerate(reversed(st.session_state.prediction_history[-5:])):
                with st.container():
                    st.write(f"**{pred['filename']}**")
                    st.write(f"🎯 {pred['prediction']}")
                    st.write(f"📊 {pred['confidence']:.1%}")
                    st.write(f"⏰ {pred['timestamp'].strftime('%H:%M:%S')}")
                    st.divider()
    
    # Main content
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 📁 Upload Your Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'],
            help="Upload any image to classify. The AI works best with clear, well-lit photos."
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)
            
            # Image info
            with st.expander("📋 Image Details", expanded=False):
                st.write(f"**Filename:** {uploaded_file.name}")
                st.write(f"**Size:** {uploaded_file.size / 1024:.1f} KB")
                st.write(f"**Dimensions:** {image.size[0]} × {image.size[1]} pixels")
                st.write(f"**Format:** {image.format}")
                st.write(f"**Mode:** {image.mode}")
            
            # Predict button
            if st.button("🚀 Classify Image", type="primary", use_container_width=True):
                success, result = predict_image(uploaded_file)
                
                if success:
                    st.session_state.current_prediction = result
                else:
                    st.error(result)
    
    with col2:
        st.markdown("### 📊 Classification Results")
        
        if st.session_state.current_prediction:
            display_prediction_results(st.session_state.current_prediction)
        else:
            st.info("👈 Upload an image and click 'Classify Image' to see results here!")
            
            # Show example of what results look like
            with st.expander("👀 See Example Results", expanded=False):
                st.markdown("""
                **Example Classification Results:**
                
                🎯 **Main Prediction:** Golden Retriever  
                📊 **Confidence:** 94.2%
                
                **Top 5 Predictions:**
                1. Golden Retriever (94.2%)
                2. Labrador Retriever (3.1%)
                3. Nova Scotia Duck Tolling Retriever (1.8%)
                4. Cocker Spaniel (0.5%)
                5. Irish Setter (0.4%)
                """)
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col2:
        st.markdown("**Built with ❤️ using FastAPI + Streamlit + PyTorch**")

if __name__ == "__main__":
    main()