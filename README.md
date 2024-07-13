# Vehicle Detection in Diverse Weather Conditions using YOLOv8 and Optuna

## Project Overview

This project addresses the challenge of vehicle detection across various weather and lighting conditions using the AVD-Dataset. We employ YOLOv8, a state-of-the-art object detection model, optimized with Optuna, to achieve robust performance in detecting multiple vehicle types under diverse environmental conditions.

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
4. [Results](#results)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Future Work](#future-work)
8. [Contributing](#contributing)
9. [License](#license)

## Introduction

Vehicle detection in varying weather conditions is a crucial task for advanced driver assistance systems (ADAS) and autonomous driving. This project aims to improve detection accuracy and robustness using the AVD-Dataset, which comprises images captured under various weather conditions.

## Dataset

- **AVD-Dataset**: 3,200 images of vehicles in diverse weather conditions
- **Training set**: 2,600 images
- **Validation set**: 200 images
- **Annotation format**: YOLO format
- **Vehicle classes**: 15 (car, bike, auto, rickshaw, cycle, bus, minitruck, truck, van, taxi, motorvan, toto, train, boat, cycle van)

## Methodology

### Model Architecture

We used YOLOv8x, the latest iteration in the YOLO (You Only Look Once) family of object detection models. YOLOv8x offers state-of-the-art performance, balancing speed and accuracy for real-time object detection.

### Hyperparameter Optimization

We employed Optuna, an efficient hyperparameter optimization framework, to fine-tune our model. The optimization process focused on the following hyperparameters:

- Epochs: Range 5-10
- Batch size: Fixed at 8 due to computational constraints
- Learning rate: Log-uniform distribution between 1e-5 and 1e-2
- Image size: Fixed at 720x720 pixels

The objective of the optimization was to maximize the mean Average Precision (mAP50-95) metric.

### Training Process

1. Initialized YOLOv8x with pre-trained weights
2. Conducted 5 optimization trials using Optuna
3. For each trial:
   - Fine-tuned the model on the AVD-Dataset with suggested hyperparameters
   - Evaluated performance on the validation set
4. Selected the best trial based on the highest mAP50-95 score

## Results

### Optimal Hyperparameters

- Epochs: 8
- Batch size: 8
- Learning rate: 2.645620762943047e-05
- Image size: 720x720 pixels

### Performance Metrics

- mAP50-95 (B): 0.292357 (29.24%)
- mAP50 (B): 0.5608 (56.08%)
- Precision (B): 0.5981 (59.81%)
- Recall (B): 0.5020 (50.20%)
- Fitness score: 0.3192 (combined metric)

### Class-wise Performance (mAP)

1. Taxi: 62.02%
2. Bike: 43.61%
3. Car: 41.05%
4. Bus: 39.28%
5. Truck: 30.59%
6. Rickshaw: 29.24%
7. Motorvan: 29.24%
8. Train: 29.24%
9. Boat: 29.24%
10. Auto: 21.81%
11. Toto: 21.06%
12. Minitruck: 15.42%
13. Van: 9.71%
14. Cycle van: 29.24%
15. Cycle: 7.83%

### Processing Speed

- Preprocess: 0.26 ms/image
- Inference: 24.61 ms/image
- Loss computation: 0.0015 ms/image
- Postprocess: 1.31 ms/image
- Total: ~26.18 ms/image (potential for ~38 FPS)

## Installation

To set up the project environment:

1. Clone this repository:

git clone https://github.com/your-username/vehicle-detection-yolov8.git
cd vehicle-detection-yolov8

2. Install the required packages:

pip install ultralytics optuna pyyaml

## Usage

To train the model and reproduce the results:

1. Prepare your dataset in the YOLO format and update the `data_path` in the script.

2. Run the optimization script:

python train_optimize.py

3. After optimization, train the model with the best hyperparameters:

python train_best.py

4. For inference on new images:

python inference.py --input path/to/image --weights path/to/best_weights.pt

## Future Work

1. Implement advanced data augmentation techniques to enhance model robustness to weather variations.
2. Explore architectural modifications to improve detection of challenging classes (e.g., cycles, vans).
3. Investigate ensemble methods or model distillation to boost overall performance.
4. Conduct error analysis to identify specific weather conditions or vehicle types that need targeted improvements.

## Contributing

We welcome contributions to improve the project. Please follow these steps:

1. Fork the repository
2. Create your feature branch 
3. Commit your changes
4. Push to the branch
5. Open a Pull Request
