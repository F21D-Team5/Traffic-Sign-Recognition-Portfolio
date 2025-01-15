# Traffic Sign Recognition Portfolio

## Description:
This project is a part of the "Data Mining and Machine Learning Portfolio" coursework during my Master's program. It involves building a machine learning model to classify German traffic signs based on the German Street Sign Recognition Benchmark (GTSRB) dataset. The subset provided contains greyscale images resized to 48x48 pixels, representing 10 traffic sign categories.

## Dataset Details
Original Dataset: [German Street Sign Recognition Benchmark (GTSRB)](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)

Subset Used:
Number of Classes: 10
Number of Images: 9690
Image Format: Greyscale, 48x48 pixels
Features: Each row contains 2304 features (flattened image) and 1 class label.
Class Labels and Categories:
Label	Traffic Sign
0	Speed Limit 20
1	Speed Limit 30
2	Speed Limit 50
3	Speed Limit 60
4	Speed Limit 70
5	Left Turn
6	Right Turn
7	Beware Pedestrian Crossing
8	Beware Children
9	Beware Cycle Route Ahead

## Project Objectives
1. Explore the provided dataset and understand the data structure.
2. Preprocess the data to ensure it is suitable for machine learning tasks.
3. Build and evaluate machine learning models to classify traffic signs.
4. Apply data mining techniques to gain insights into the dataset.

## Methodology
#### 1. Import Libraries: 
Common libraries such as NumPy, pandas, and OpenCV were imported, along with matplotlib for plotting and Scikit-learn for data mining and analysis.

#### 2. Data Preprocessing:
The training dataset (X_train and Y_train) was read, and feature selection was performed using Principal Component Analysis (PCA).
Pixel values were normalized from [0-255] to [0-1] by dividing by 255 to enhance model stability.

#### 3. Feature Engineering:
New datasets with 50, 100, and 200 features were created, each corresponding to specific classes for classification.

#### 4. Modeling:
1. Naïve Bayes Classifier: The model was trained on the datasets, achieving accuracies of 0.73, 0.74, and 0.66 for 50, 100, and 200 features, respectively.
2. K-Means Clustering: Feature scaling was performed using Standard Scaler, followed by PCA for dimensionality reduction. The optimal number of clusters was determined using the elbow and silhouette methods, resulting in a K value of 4.
3. Decision Trees: A decision tree classifier was plotted with a maximum depth of 2. Ten-fold cross-validation was used to evaluate the model, with metrics such as true positive rate (TPR), false positive rate (FPR), precision, and recall computed for each fold.
4. Random Forest Classifier: Implemented using the Weka library, following similar methods as the decision tree.
5. Convolutional Neural Network (CNN): A suitable CNN model for image classification was designed, experimenting with the number of convolutional layers and kernel sizes. The model was compiled using the Adam optimizer and tuned using Grid Search and Randomized Search to identify the best hyperparameters.

#### 5. Evaluation:
The final model was evaluated on the test datasets, achieving a test accuracy of 92%.

## Conclusion
The project demonstrated that supervised learning algorithms, particularly the CNN, significantly improved the accuracy of traffic sign classification. Further optimizations and testing could enhance model performance.

### Acknowledgments
This project was completed as a group effort during the Data Mining and Machine Learning course.
We would like to thank our teammates for their collaboration and contributions, which were instrumental in the successful completion of this project.



