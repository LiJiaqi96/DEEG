<div align=center><img width="900" height="150" src="/figures/logo.png">  

# DEEG

A Python toolkit for EEG data processing and Deep Learning.

## Structure

The structure of this package contains four main layers for creating a novel workflow in EEG data analysis:
1. __Data Access Layer__
    * Creates APIs for existing EEG datasets (DEAP, SEED, etc.)
    * Preprocess raw data file format (mat, csv, txt, etc.)
    
2. __Data Process Layer__
    * Data Processing Functionalities including data quality checking, null/nan values checking and noise evaluation
    * Signal Processing Functionalities of filtering & enhancement designed for EEG signal data
    
3. __Data Extraction Layer__
    * Feature Engineering Functionalities including:
        * Temporal
        * Frequency
        * Multi-channel
        * Deep Learning Features Extraction
    
4. __Data Application Layer__
    * Machine Learning/Deep Learning platform
    * Other Applications (Word Cloud, Visualizations, etc.)
   
These four layers work independently to each other and provide fundamental usages on EEG data analysis and modelling.
