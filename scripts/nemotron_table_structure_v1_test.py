import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from nemotron_table_structure_v1.model import define_model
from nemotron_table_structure_v1.utils import plot_sample, postprocess_preds_table_structure, reformat_for_plotting

# Load image
path = "./example.png"
img = Image.open(path).convert("RGB")
img = np.array(img)

# Load model
model = define_model("table_structure_v1")

# Inference
with torch.inference_mode():
    x = model.preprocess(img)
    preds = model(x, img.shape)[0]

# Post-processing
boxes, labels, scores = postprocess_preds_table_structure(preds, model.threshold, model.labels)

# Plot
boxes_plot, confs = reformat_for_plotting(boxes, labels, scores, img.shape, model.num_classes)

plt.figure(figsize=(30, 15))
for i in range(1, 4):
    boxes_plot_c = [b if j == i else [] for j, b in enumerate(boxes_plot)]
    confs_c = [c if j == i else [] for j, c in enumerate(confs)]

    plt.subplot(1, 3, i)
    plt.title(model.labels[i])
    plot_sample(img, boxes_plot_c, confs_c, labels=model.labels, show_text=False)
plt.show()
