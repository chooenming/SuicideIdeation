# Machine Learning in Online Health Monitoring: Suicidal Ideation Detection

## Project Overview
This repository contains the code for my final year project on using machine learning for online health monitoring, specifically focusing on the detection of suicidal ideation. The project consists of two main parts: data cleaning and exploratory data analysis (EDA).

## Project Structure
- **01_Data_Cleaning.ipynb**: Jupyter Notebook containing functions and scripts for data cleaning.
- **02_EDA.ipynb**: Jupyter Notebook containing functions and scripts for exploratory data analysis.
- **utils.py**: Python file containing utility functions required for data cleaning and EDA.
- **app.py**: Python script for deploying the best model using Streamlit.
- **MDS_FYP_ALBert.ipynb**: Jupyter Notebook containing functions and scripts for ALBERT model
- **MDS_FYP_DistilBERT.ipynb**: Jupyter Notebook containing functions and scripts for DistilBERT model
- **MDS_FYP_RoBERTa.ipynb**: Jupyter Notebook containing functions and scripts for RoBERTa model

## Environment
All experiments and development work were conducted on Google Colab due to the use of GPU.

## Deployment
After conducting experiments and selecting the best model, it will be deployed using Streamlit. The deployment code can be found in `app.py`.