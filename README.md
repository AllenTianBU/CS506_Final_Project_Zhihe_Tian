# Happiness Predictor Trained on GSS

## How to Build and Run the Code

### 1. Download the Data
The GSS dataset is too large to store in this repository (1.9GB before cleaning). So please download it using the link below! 
1. https://www.kaggle.com/datasets/norc/general-social-survey?resource=download
2. Download the CSV and move the file to: `GSS_Data_CSV_CodeBook/gss.csv`

### 2. Install Dependencies
I have followed the standard procedures, so it should be fairly easy to test the code using the command lines below.
```bash
make install
```

### 3. Run the Model
```bash
make run
```

### 4. Run Tests
```bash
make test
```

## Methods
### Data collection and process
Justify why this source
Shown in code
### Data cleaning
Describe in detail how I did it
How I handled missing, noisy data
Shown in code
### Feature extraction
Define features clearly and explained
Extract features appropriate for this task
### Model training & Evaluation
Train procedure
Why linear regression (the particular type?)
Justify my strategy
Discussion of failure
## Results/Visualization
Plots here and talk about results
