# Denmark Housing Prices Prediction using PyTorch

This project implements a deep learning regression framework to predict real estate prices in Denmark. By utilizing a dataset of 100,000 transactions, the system analyzes the relationship between property attributes, geographical data, and macroeconomic indicators to provide accurate price estimations.

## Problem Description
Real estate valuation is a multi-faceted regression problem. Factors such as location, square footage, and building year interact with broader economic trends like inflation and mortgage interest rates. This project leverages neural networks to capture these non-linear dependencies, offering a robust tool for automated property appraisal.

## Dataset Information
The project utilizes the **DK Housing Prices Sample 100k** dataset.
- **Source:** Kaggle (Denmark Housing Prices dataset).
- **Key Features:** House type, sales type, year of construction, number of rooms, square meters (sqm), city, region, and annual inflation rates.
- **Target:** The `purchase_price` variable.

## Technical Comparison: ReLU vs. Sigmoid
This study compares different activation functions to evaluate their impact on model convergence and accuracy:

| Feature | ReLU Model (1024 Neurons) | Sigmoid Model (512 Neurons) |
| :--- | :--- | :--- |
| **Output Range** | 0 to infinity | 0 to 1 |
| **Gradient Flow** | Prevents saturation in the positive domain, reducing vanishing gradient issues. | Suffers from vanishing gradients as input values increase or decrease. |
| **Efficiency** | Highly computational efficient due to simple thresholding logic. | Computationally expensive due to exponential function requirements. |
| **Usage** | Recommended as the default for hidden layers in deep architectures. | Historically significant but primarily used now for output probability scaling. |



## Project Instructions

### Prerequisites
The environment requires Python and the following libraries:
- pandas, numpy, matplotlib, scikit-learn, and torch.

### Setup and Execution
1. Place the dataset file `DKHousingPricesSample100k.csv` in the directory: `D:\Neural Networks\`.
2. Execute the primary script: `python main.py`.

### Model Checkpoints
The training loop identifies the state with the lowest validation loss and exports the model weights as `.pth` files to the `D:\Neural Networks\` directory. Separate files are generated for ELU, ReLU, and Sigmoid configurations to allow for modular inference and further evaluation.

---
Developed using PyTorch and Scikit-Learn.
