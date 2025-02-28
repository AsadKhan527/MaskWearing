import pandas as pd
import matplotlib.pyplot as plt

# Path to the YOLOv5 results.csv file
results_file = "/Users/asadkhan/Desktop/Lab06/yolov5/runs/train/exp3/results.csv"

# Load the training results
df = pd.read_csv(results_file)
print(df.columns.tolist())
df['               epoch']
# Strip all leading and trailing whitespaces from column names
df.columns = df.columns.str.strip()

# Double-check column names after stripping
print(df.columns.tolist())


# Plot settings
plt.figure(figsize=(15, 10))

# Plot training and validation losses
plt.subplot(2, 2, 1)
plt.plot(df['epoch'], df['train/box_loss'], label='Box Loss', color='blue')
plt.plot(df['epoch'], df['train/obj_loss'], label='Object Loss', color='green')
plt.plot(df['epoch'], df['train/cls_loss'], label='Class Loss', color='red')
plt.title('Training Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot Precision and Recall
plt.subplot(2, 2, 2)
plt.plot(df['epoch'], df['metrics/precision'], label='Precision', color='blue')
plt.plot(df['epoch'], df['metrics/recall'], label='Recall', color='green')
plt.title('Precision and Recall')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

# Plot mAP@0.5 and mAP@0.5:0.95
plt.subplot(2, 2, 3)
plt.plot(df['epoch'], df['metrics/mAP_0.5'], label='mAP@0.5', color='blue')
plt.plot(df['epoch'], df['metrics/mAP_0.5:0.95'], label='mAP@0.5:0.95', color='green')
plt.title('Mean Average Precision (mAP)')
plt.xlabel('Epoch')
plt.ylabel('mAP')
plt.legend()
plt.grid(True)

# Plot learning rate
plt.subplot(2, 2, 4)
plt.plot(df['epoch'], df['x/lr0'], label='Learning Rate', color='purple')
plt.title('Learning Rate')
plt.xlabel('Epoch')
plt.ylabel('LR')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
