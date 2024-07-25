# nyc_taxi_trip_duration
This a repo for solving the problem of NYC taxi trip duration (kaggle problem). 
Jupyter notebook is used for analysis and python files for modeling. 
Also used docker for setup.

### !! Note !!
  - The data I used In my notebook is private but, you can follow the notebook with the [data](https://www.kaggle.com/datasets/yasserh/nyc-taxi-trip-duration)


### How to run the modeleling.py file
- First setup the dockerfiles.
- Then run the container. the container will run the code and print the results.
    
      - docker build -t nyt/v1 .
      - docker container run --name nyt_cont  nyt/v1

### !! files 
  - [Dockerfile](https://github.com/zynabsmaan/nyc_taxi_trip_duration/blob/main/Dockerfile): contains all necessary packages to run the modeling file.
  - [modeling.py](https://github.com/zynabsmaan/nyc_taxi_trip_duration/blob/main/modeling.py): contains the code for preprocessing the data and then run the model. 
  - [nyc](https://github.com/zynabsmaan/nyc_taxi_trip_duration/blob/main/nyc.ipynb): contains all analysis
    steps [Exploring the data and removing the outliers].




 
