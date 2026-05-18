🏠 Denmark Housing Prices Prediction (Neural Networks)

This project aims to build a housing price prediction model for real estate in Denmark using Deep Learning with the PyTorch library. The project analyzes the impact of different Activation Functions on the model’s accuracy and efficiency.

📋 Project Overview
The dataset contains 100,000 housing sales records in Denmark, combined with macroeconomic factors such as inflation rates and mortgage bond interest rates to improve prediction accuracy for the target variable purchase_price.

🛠️ Project Contributions

Data Preprocessing
Cleaned the dataset and removed missing values.
Applied StandardScaler for numerical feature scaling.
Used OneHotEncoder for categorical features such as city, house_type, and region.
Model Architecture
Designed deep neural network models with hidden layers containing 1024 and 512 neurons.
Implemented BatchNorm1d to stabilize training and Dropout to reduce overfitting.
Activation Functions Comparison

Built and trained four different models to compare the performance of:

ReLU
Sigmoid
Tanh
ELU
Training & Evaluation System
Developed a training pipeline that automatically saves the best model based on the lowest validation loss.
Used ReduceLROnPlateau to reduce the learning rate when performance stops improving.

📊 Activation Functions Comparison

Activation Function	Output Range	Advantages in This Project
ReLU	[0, ∞)	Fastest learning and most efficient for deep layers.
Sigmoid	[0, 1]	Faced challenges due to the vanishing gradient problem.
Tanh	[-1, 1]	More stable performance because outputs are centered around zero.
ELU	(-α, ∞)	Handled negative values smoothly and reduced dead neuron issues.

🚀 Setup & Execution

Make sure the required libraries are installed:

pip install torch pandas scikit-learn matplotlib numpy

Place the dataset file DKHousingPricesSample100k.csv in:

D:\Neural Networks\

Then run the Jupyter Notebook or Python script.

📉 Results

R² Score was used as the main evaluation metric.
The target goal was to achieve an accuracy score of 0.6 or higher.
ReLU and ELU achieved the best overall performance in terms of convergence speed and lower validation loss.
