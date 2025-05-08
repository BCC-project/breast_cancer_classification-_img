# Breast Cancer Classification from Histopathological Images

This repository presents a project focused on the classification of breast ultrasound images into three distinct categories: **benign**, **malignant**, and **normal**. The primary objective is to accurately detect the presence or absence of breast cancer through advanced image analysis.

The classification model is built upon state-of-the-art deep learning methodologies, specifically leveraging **Convolutional Neural Networks (CNNs)** and **Transfer Learning**. The architecture employs **DenseNet121** as the base model, trained on a comprehensive dataset of annotated breast ultrasound images to enhance diagnostic performance.

Transfer learning is utilized by freezing the initial layers of DenseNet121—pre-trained on a large, diverse dataset—to retain generalized feature extraction capabilities. The later layers are fine-tuned to capture domain-specific patterns relevant to breast ultrasound imagery.

This work constitutes a **multiclass classification** task, aiming to categorize input images into one of the three defined classes, thereby contributing to improved accuracy and efficiency in breast cancer detection.

