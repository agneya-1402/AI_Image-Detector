# AI Image Detector

A Streamlit-based web application that detects whether an image is AI-generated or real using deep learning. The detector uses a fine-tuned ResNet18 model to analyze images and provide confidence scores.
[dashboard.png](https://github.com/agneya-1402/AI_Image-Detector/blob/main/dashboard.png)
## 🌟 Features

- Upload images through the web interface
- Real-time analysis of images
- Visual confidence score display
- Detailed analysis breakdown
- Interactive gauge chart for confidence visualization

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/agneya-1402/ai_image-detector.git
cd ai_image-detector
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## 💻 Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in your terminal (typically http://localhost:8501)

3. Use the application by either:
   - Uploading an image file through the file uploader
   - Pasting an image URL (feature coming soon)

## 📁 Project Structure

```
ai-image-detector/
├── app.py              # Streamlit web application
├── main_3.py          # AI Image Detector model implementation
├── requirements.txt    # Project dependencies
├── README.md          # Project documentation
└── imgs/              # Sample images directory
```

## 🤖 Model Details

The detector uses a modified ResNet18 architecture with:
- Custom classification head for binary prediction
- Image preprocessing and augmentation
- Transfer learning from pretrained weights
- Binary classification (Real vs AI-Generated)

## 📊 Results Interpretation

- **Green Box**: Indicates the image is likely real
- **Red Box**: Indicates the image is likely AI-generated
- **Confidence Gauge**: Shows the model's confidence level (0-100%)
- **Detailed Analysis**: Provides specific indicators that influenced the prediction

## ⚙️ Requirements

- Python 3.8+
- CUDA-capable GPU (optional, for faster processing)
- See requirements.txt for complete list of dependencies

## 🛠️ Development

To contribute to this project:

1. Fork the repository
2. Create a new branch for your feature
3. Make your changes
4. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## ⚠️ Known Issues

- Large images may require additional processing time

## 📧 Contact

For questions and support, please open an issue in the GitHub repository.

## 🙏 Acknowledgments

- Built with Streamlit
- Uses PyTorch and torchvision
- ResNet architecture by Microsoft Research
