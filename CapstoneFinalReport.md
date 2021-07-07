{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "13fc2569",
   "metadata": {},
   "source": [
    "# Food Image Classification\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1efb50fa",
   "metadata": {},
   "source": [
    "## Problem Identification \n",
    "\n",
    "They say we eat with our eyes and our noses first, before we actually taste food. For humans, recognizing different food items with specific ingredients is a fairly simple task, but with so many unique, traditional, cultural, and location specific dishes, it could be helpful if the food could be recognized by machines. This could also help in calorie counting or for dietary restrictions. Imagine being able to take a picture of your meal and having enough information to understand the calories and macros involved. The first step to achieving this is food image classification. In this project, I work with a dataset consisting of over 101 food categories, with over 1000 images, with the goal of building a machine learning model that can classify the images into their correct categories. \n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e17f79ba",
   "metadata": {},
   "source": [
    "## Data Collection and Organization\n",
    "\n",
    "The data for this project is sourced from Kaggle, the Food Images(Food-101) dataset. There are multiple groups of images, already split into training and testing sets in hdf5 format, based on the image sizes. I chose to use the 384 x 384 x 3 for the training set, and the 128 x 128 x 3 for the testing set. Meta data including the classes and labels were used as well. \n",
    "\n",
    "![training images](./images/training.png)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "11fd8cec",
   "metadata": {},
   "source": [
    "## Exploratory Data Analysis \n",
    "\n",
    "Since the data are images, I first took a quick peak at the images, and then built a dataframe with the correct labels and number of each image class. Upon looking closer, it is clear that there is an imbalance of data throughout the classes. \n",
    "\n",
    "![training images imbalance](./images/classes_training.png)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0b9e5c47",
   "metadata": {},
   "source": [
    "## Data Preprocessing\n",
    "\n",
    "After reshaping the training and testing images to be 128x128x3 in size for the models later on, we normalize the data by subtracting the training mean from both data sets and then dividing by the training standard deviation. This will center the cloud of data around points 0,0 on the graph, and make data dimentions aproximately on the same scale, allowing model training to run faster. "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ead008a1",
   "metadata": {},
   "source": [
    "## Modeling and Metrics \n",
    "\n",
    "#### Created CNN model from scratch\n",
    "\n",
    "To get a baseline understanding, I first created a CNN model from scratch. It has 3 convolution layers followed by 2 dense layers. I implemented an early stopping monitor which will select the best model based off of the validation loss metric. The loss metric measures how confident we are in the prediction. This typically prevents overfitting compared to measuring the accuracy metric. Unfortunately, the results weren't that great, as loss continued to increase. The best scored model has a 0.02 accuracy. \n",
    "\n",
    "#### Transfer Learning with VGG16\n",
    "\n",
    "Next, I tried using transfer learning with VGG16, as a model with thousands of pretrained categories, it is expected to perform better than the first CNN model, that is trained on less than 1000 images. I allowed the last 4 layers to be tuned, and added 3 dense layers. Although the training accuracy here increase compared to model 1, the validation accuracy was still stuck on 0.02. This is a clear sign of overfitting when training accuracy increases, but validation accuracy does not, and validation loss increases. \n",
    "\n",
    "#### VGG16 with Augmented Images\n",
    "\n",
    "For a final model, I applied image augmentations to see if this would help increase validation accuracy while keeping validation loss low. I used ImageDataGenerator to augment the images while the model is training. The validation accuracy score here did increase to 0.11. This is good news, as it tells us that with additional training images, or augmented images, we are seeing better results. \n",
    "\n",
    "![model 3 accuracy](./images/model3accuracy.png)\n",
    "![model 3 loss](./images/model3loss.png)\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "25667169-5c77-4acb-9004-6fd1f9ee2cff",
   "metadata": {},
   "source": [
    "## Conclusion and Future Improvements \n",
    "\n",
    "1. The largest conclusion is that we do not have enough images for each class to properly train the model. Some of the classes only consist of 5 images. Even with the addition of transfer learning, where the model is pretrained on thousands of classes already, it appears that the specific classes we are looking to predict for does not have enough training data to accurately classify and predict new images. Because of this, the primary suggestion would be to collect additional data for training before trying other further improvements.   \n",
    "\n",
    "2. It seems clear that adding augmentations has improved model perfermance, so it could make sense to add additional augmentations. \n",
    "\n",
    "3. Additional regularization could help improve the model performance. For future improvements, it might make sense to increase the L1 and L2 regularization, or consider experimenting with different drop out rates for the model.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ef1122d0-31d2-4038-a53f-50680dc3bda0",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
