🛡️ Mask Wearing Detection using YOLOv5
📌 Project Overview
Developed an image classification and object detection model to identify whether individuals are wearing face masks using the YOLOv5 deep learning framework. The model is trained on labeled images to distinguish between masked and unmasked faces, aiding in public safety enforcement and pandemic-related monitoring systems.

🔧 Key Features

Utilized YOLOv5 for real-time object detection and classification.

Automated data preprocessing and training visualization using pandas and matplotlib.

Implemented loss monitoring (box, object, and classification loss) and performance metrics tracking (Precision, Recall, mAP).

Created insightful training plots to visualize learning progress and overfitting trends.

📂 Dataset
The training was conducted on a custom image dataset of individuals with and without masks. Images were annotated using standard bounding box formats suitable for YOLOv5 training.

📊 Final Model Performance (Epoch 49):

✅ Approximate Accuracy: 58.34%

🎯 Precision: 40.55%

🔁 Recall: 76.13%

📈 mAP@0.5: 50.93%

📉 mAP@0.5:0.95: 33.09%

🖼️ Visualization
Generated comprehensive plots of:

Training losses (box, object, class)

Precision vs. Recall over epochs

Mean Average Precision (mAP) progression

Learning rate schedule

🚀 Tools & Technologies
Python · YOLOv5 · PyTorch · OpenCV · Matplotlib · Pandas

📌 Future Work

Incorporate real-time webcam detection with alert systems

Train on larger, more diverse datasets for better generalization

Integrate with edge devices for on-the-fly mask detection

