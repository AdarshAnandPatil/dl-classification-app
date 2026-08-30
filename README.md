# Deep Learning Classification App

A Flask-based web application for image classification using deep learning models. This application supports multiple classification tasks including fruit, flower, animal, handwritten letter, traffic sign, and waste classification.

## Features

- **Multi-Model Classification**: Support for 6 different classification models
- **Web Interface**: User-friendly Flask web interface
- **Image Upload**: Easy image upload and classification
- **Dashboard**: View statistics and analytics
- **Responsive Design**: Mobile-friendly UI

## Project Structure

```
dl-classification-app/
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── .gitignore                      # Git ignore rules
│
├── models/                         # Pre-trained models
│   └── README.txt
│
├── training/                       # Training scripts
│   ├── README.md
│   ├── train_image_classifier.py
│   ├── train_letter_classifier.py
│   └── model_integration.py
│
├── templates/                      # HTML templates
│   ├── base.html                  # Base template
│   ├── home.html                  # Home page
│   ├── dashboard.html             # Dashboard page
│   └── classifier.html            # Classifier page
│
├── static/                         # Static files
│   └── style.css                  # CSS styling
│
└── uploads/                        # Uploaded images
    └── .gitkeep
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AdarshAnandPatil/dl-classification-app.git
cd dl-classification-app
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. Select a classifier and upload an image to get predictions

## Training Models

To train new models, run the training scripts in the `training/` directory:

```bash
# Train image classifier
python training/train_image_classifier.py

# Train letter classifier
python training/train_letter_classifier.py

# Integrate models
python training/model_integration.py
```

## Classification Categories

1. **Fruit Classification** - Identify different types of fruits
2. **Flower Classification** - Identify different types of flowers
3. **Animal Classification** - Identify different types of animals
4. **Letter Classification** - Recognize handwritten letters
5. **Traffic Sign Classification** - Identify traffic signs
6. **Waste Classification** - Classify waste materials

## Technologies Used

- **Backend**: Flask, Python
- **Deep Learning**: TensorFlow, Keras
- **Frontend**: HTML, CSS, JavaScript
- **Data Processing**: NumPy, Pandas, Pillow
- **Machine Learning**: scikit-learn

## Requirements

- Python 3.7+
- Flask 2.3.0
- TensorFlow 2.12.0
- Other dependencies listed in requirements.txt

## Configuration

Edit `app.py` to modify:
- `UPLOAD_FOLDER` - Directory for uploaded images
- `MAX_CONTENT_LENGTH` - Maximum file upload size
- Port and host settings

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Author

Adarsh Anand Patil

## Support

For support, open an issue on the GitHub repository.
