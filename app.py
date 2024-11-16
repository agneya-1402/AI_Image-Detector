import streamlit as st
import plotly.graph_objects as go
from PIL import Image
import io
import base64
from main_3 import AIImageDetector
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="AI Image Detector",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .real {
        background-color: rgba(0, 255, 0, 0.1);
        border: 2px solid #00ff00;
    }
    .fake {
        background-color: rgba(255, 0, 0, 0.1);
        border: 2px solid #ff0000;
    }
    .stProgress > div > div > div > div {
        background-color: #00ff00;
    }
    </style>
""", unsafe_allow_html=True)

# gauge chart
def create_confidence_gauge(confidence, prediction):
    color = 'green' if prediction == 'Real' else 'red'
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confidence,
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 100], 'color': "gray"}
            ]
        },
        title = {'text': "Confidence Level"}
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=50, b=10),
        font={'size': 16}
    )
    
    return fig

def main():
    # Header
    st.title("🔍 AI Generated Image Detector")
    st.markdown("Upload an image to check if it's AI-generated or real.")
    
    # Initialize detector
    detector = AIImageDetector()
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])
    
        
    col1, col2 = st.columns([1, 1])
    
    if uploaded_file is not None :
        try:
            # Process the image
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                # Save temporarily
                temp_path = "temp_image.jpg"
                image.save(temp_path)
            else:
                # Handle URL input (you'll need to implement this part)
                st.error("URL functionality is not implemented yet")
                return
            
            # Make prediction
            prediction, confidence = detector.predict(temp_path)
            
            with col1:
                st.subheader("Uploaded Image")
                st.image(image, use_column_width=True)
            
            with col2:
                st.subheader("Analysis Results")
                
                # Display result with colored box
                result_class = "real" if prediction == "Real" else "fake"
                st.markdown(f"""
                    <div class="result-box {result_class}">
                        <h3>Prediction: {prediction}</h3>
                        <h4>Confidence: {confidence:.2f}%</h4>
                    </div>
                """, unsafe_allow_html=True)
                
                # Display confidence gauge
                st.plotly_chart(create_confidence_gauge(confidence, prediction))
                
                # Additional information
                st.markdown("### Details")
                if prediction == "Real":
                    st.success("""
                        This image appears to be a real photograph. Key indicators:
                        - Natural lighting and shadows
                        - Consistent details and textures
                        - Realistic imperfections
                    """)
                else:
                    st.error("""
                        This image appears to be AI-generated. Common indicators:
                        - Unusual artifacts or distortions
                        - Inconsistent details
                        - Perfect symmetry or patterns
                    """)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()