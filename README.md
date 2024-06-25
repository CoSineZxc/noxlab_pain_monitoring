# Online pain monitoring study
We aim to characterize how back pain fluctuates over the scale of a few minutes by designing a continuous pain-tracking task for patients with chronic back pain. The pain tracking task involves the continuous rating of perceived intensity and a sparse sampling of pain predictions and confidence levels. The project can potentially help to better phenotype patients, which could be treated with a precision medicine approach and shed light on how to improve the quality of life of patients. The task we develop also has the potential to be incorporated into NHS digital healthcare in the future.

Pilot study 1: 28th January 2023

Pilot study 2: 03rd Feb 2023

Pilot study 3a, 3b: 11th Feb 2023

Pilot study 3c, 3d: 12th Feb 2023


## How to run the python analysis pipeline
In each folder, there is a ```Data``` folder and a ```Code``` Folder. The ```Data``` Folder contains the ```.csv``` files in the study batch while the ```Code``` Folder consists of a Jupyter Notebook that performs <strong>Descriptive Analysis</stsrong> on the data. 

Packages needed: NumPy, Matplotlib, Pandas, SciPy, ordPy, statsmodels.api, os

On top of the Jupyter notebook, the functions for pre-processing (Windowing and low pass filtering) is shown, followed by the analysis functions of common statistical parameters, frequency, permutation entropy and autocorrelation.

Below the functions, the prediction and confidence data is plotted where if > 20% of the data has a response time of > 10 seeconds, the data will be ```NaN```, the participant will be removed from the trial. Below that, the continuous pain trace will be plotted.

After this, preprocessing (window and low pass filter) will be applied

Other analysis functions and will also be applied after this.

## The purpose of each folder
+ screening:
    Plot the distribution of scores in MSK-HQ for all screened participants. 
+ pilot-analysis-1~5:
    Before the pain monitoring experiment, we designed 5 versions of the experiment. 
+ actual-analysis-first-10:
    After all of the pilot version, we tried the final Pain Monitoring in 10 participants.
+ actual-analysis-first~second:
    During the pain monitoring experiment, we use these script for data quality check and exclude participants
+ data_preprocess_xz:
    After data collection, here is the final script for data preprocessing
+ data_analysis_xz:
    Part II of data preprocessing, for split data and exclude bad trials
+ data_analysis_normality_xz:
    normality check for rating data
+ final-analysis:
    Sharon's data analysis