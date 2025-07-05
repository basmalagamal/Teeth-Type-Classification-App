# **Teeth-Type-Classification-App**
Project Overview
This project develops a deep learning pipeline to classify dental images into 7 distinct categories using convolutional neural networks. It utilizes a fine-tuned EfficientNetB0 model pretrained on ImageNet, adapted to classify teeth images with high accuracy.

The pipeline includes:
  - Data preprocessing and augmentation on RGB dental images.
  - Training and fine-tuning EfficientNetB0 on the teeth dataset.
  - Evaluation with accuracy metrics, confusion matrix, and training curves.
  - Deployment of the model as an interactive web app using Streamlit.
-------------------------------------------------------------------------------------------------

**Dataset Description**
The dataset contains RGB .jpg images organized into 7 classes representing different teeth conditions or types:
[ CaS, CoS, Gum, MC, OC, OLP, OT ]

Teeth_Dataset/
├── Training/
│   ├── CaS/
│   ├── CoS/
│   ├── Gum/
│   ├── MC/
│   ├── OC/
│   ├── OLP/
│   └── OT/
├── Validation/
│   └── (same structure)
└── Testing/
    └── (same structure)

-----------------------------------------------------------------------------------------------

**Model Architecture**
Base model: EfficientNetB0 pretrained on ImageNet (without top layers).
Custom classification head added:
  - GlobalAveragePooling2D
  - Batch Normalization
  - Dropout layers to reduce overfitting
  - Dense layer with softmax activation for 7 classes.

Model training includes:
  - Initial frozen base model training.
  - Fine-tuning top layers of EfficientNetB0 with lower learning rate.

-------------------------------------------------------------------------------------------------

**Project Structure**

teeth-classification-app/
│
├── app/
│   ├── app.py                  # Streamlit app entry point
│   ├── utils.py                # Helper functions (preprocessing, prediction)
│   └── model/
│       └── efficientnet_model.h5  # Trained model file
│
├── data/
│   └── example_teeth.jpg       # Sample test images
│
├── notebooks/
│   └── training_notebook.ipynb # Model training and EDA notebook
│
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation (this file)
├── .gitignore
└── LICENSE

**How to run the App**

1. Clone the repo:
  git clone https://github.com/yourusername/teeth-classification-app.git
  cd teeth-classification-app
2. Install dependencies:
  pip install -r requirements.txt
3. Run the Streamlit app:
  streamlit run app/app.py
  Use the app:

4. Upload a dental image (RGB .jpg).
  The app preprocesses the image and predicts its class.
  Displays predicted label and confidence.

