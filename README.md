# Emotion Classification using Naive Bayes
## Project Overview
This project implements a Naive Bayes classifier to predict emotions from textual data. The model is built using the scikit-learn library, with text preprocessing handled by CountVectorizer. A confusion matrix is generated to evaluate the model's performance, and the project also includes functionality to predict the emotion of a new user-inputted comment.

## Requirements
* Python 3.x
* pandas
* scikit-learn
* seaborn
* matplotlib
## Installation
You can install the required packages using pip:

'''bash
pip install pandas scikit-learn seaborn matplotlib
## Dataset
The dataset used is a CSV file containing two columns:

* Comment: The textual data to be analyzed.
* Emotion: The corresponding emotion label for each comment.
Make sure to have the dataset file at the specified path or update the dataset variable in the script to reflect the correct location.

## Running the Project
* Ensure all dependencies are installed.
* Place the dataset in the specified location or modify the path in the code.
* Run the Python script:
bash
python emotion_classifier.py
* After the script runs, it will display:
* Model Accuracy
* Confusion Matrix as a heatmap
* Prompt for a new comment input, which the model will classify.
* Enter a comment when prompted, and the model will predict the associated emotion.

## Example Usage
bash
Enter something: I am so excited for the new project!
Predicted Emotion: ['joy']
## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page if you want to contribute.
