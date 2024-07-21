
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import pandas as pd

from math import cos, sin, pi, acos
import argparse
import logging


version = 1
logging.basicConfig(filename=f'perprocessing_modeling_info{version}.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)


class FeatureExtracting(object):
    def __init__(self, df):
        self.df = df
        self.df['pickup_datetime'] = pd.to_datetime(self.df['pickup_datetime'])
        
    
    def _cal_distance(self, row):
        """
        This function compute the distance between two points using latitude and longitude.
        returns:
        distance in kilometer

        """
        logging.info(f"claculate distance from long/lat info")
        pickup_lat, pickup_lon, dropoff_lat, dropoff_lon = row['pickup_latitude'], row['pickup_longitude'], row['dropoff_latitude'], row['dropoff_longitude']

        r = 6371
        p = pi/180

        out = acos(round(sin(pickup_lat * p) * sin(dropoff_lat * p) +\
            cos(pickup_lat * p) * cos(dropoff_lat * p) *\
            cos((dropoff_lon - pickup_lon) * p), 12)) * r

        return round(out, 5) 
    
    def extr_hour(self):
        logging.info(f"Extracting hour and add feature of binned_hour")
        self.df['hour'] = self.df['pickup_datetime'].dt.hour
        self.df['binned_hour'] = pd.cut(self.df["hour"], bins=[-1, 5, 11, 17, 23], labels=[1, 2, 3, 4])
    
    def extr_workday(self):
        logging.info(f"Extracting day and add feature is_workday.")
        self.df['day_name'] = self.df['pickup_datetime'].dt.day_name().str.slice(stop=3)
        self.df['is_workday'] = self.df['day_name'].apply(lambda x: 0 if x == ('Sat' or 'Sun') else 1) 
    
    def add_long_lat_degress(self, location):
        logging.info(f'convert {location} to x, y , z.')
        self.df[f'x_{location}'] = np.cos(self.df[f"{location}_latitude"]) * np.cos(self.df[f"{location}_longitude"])
        self.df[f'y_{location}'] = np.cos(self.df[f"{location}_latitude"]) * np.sin(self.df[f"{location}_longitude"])
        self.df[f'z_{location}'] =  np.sin(self.df[f"{location}_latitude"])


    def x_y_z_distance(self):
        logging.info('adding distance x,y,z')
        diff_x = self.df['x_dropoff'] - self.df['x_pickup']
        diff_y = self.df['y_dropoff'] - self.df['y_pickup']
        diff_z = self.df['z_dropoff'] - self.df['z_pickup'] 

        distance_x_y_z = np.sqrt(np.power(np.sum([diff_x, diff_y, diff_z], axis=0), 2))
        self.df['distance_x_y_z'] = distance_x_y_z

    def extr_distance(self):
        if 'distance' not in self.df.columns:
            logging.info('Adding distance')
            self.df['distance'] = self.df.apply(lambda x: self._cal_distance(x), axis=1)
            

    
    def extr_log_10_distance(self):
        if 'log_10_distance' not in self.df.columns:
            logging.info('Adding log_10_distance')
            self.df['log_10_distance'] = np.log10(self.df.distance + 1e-1) 
        

    def extr_log_trip_duration(self):
        if 'log_trip_duration' not in self.df.columns and 'trip_duration' in self.df.columns:
            logging.info('Adding log_trip_duration')
            self.df['log_trip_duration'] = np.log1p(self.df.trip_duration)

    
    def apply_feat_extr(self):
        
        self.extr_distance()
        self.extr_hour()
        self.extr_workday()
        self.extr_log_trip_duration()
        self.extr_log_10_distance()
        self.add_long_lat_degress("pickup")
        self.add_long_lat_degress("dropoff")
        self.x_y_z_distance()
        return self.df



class FeatureTransformtion(object):

    def apply(self, df_train, df_val):
        logging.info(f"Apply transformation  like one_hot_encoding and scaling.")
        column_transformer = ColumnTransformer([
            ('hour_one_hot', OneHotEncoder(handle_unknown='ignore'), ['binned_hour']),
            ('is_workday_one_hot', OneHotEncoder(handle_unknown='ignore'), ['is_workday']),
            # ('passenger_scaling', MinMaxScaler(), ['passenger_count'])
            ]
        , remainder = 'passthrough', verbose_feature_names_out=False)
        
        column_transformer.fit(df_train)

        transformed_train = column_transformer.transform(df_train)
        transformed_val = column_transformer.transform(df_val)
        
        return transformed_train, transformed_val, column_transformer.get_feature_names_out()
        


class ReaderSaverData(object):
    def __init__(self):
        pass

    def read_data(self, file_name):
        df = pd.read_csv(file_name, index_col=False)
        return df
    
    def save_data(self, df, file_name):
        print(f'saved {file_name}')
        df.to_csv(f'clean/{file_name}.csv', index=False)


def predict_eval(model, x, y, name):
    y_train_pred = model.predict(x)
    rmse = mean_squared_error(y, y_train_pred, squared=False)
    r2 = r2_score(y, y_train_pred)
    logging.info(f"{name} RMSE = {rmse:.4f} - R2 = {r2:.4f}")

    print(f"{name} RMSE = {rmse:.4f} - R2 = {r2:.4f}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Regressors Homework')

    parser.add_argument('--data_version', type=str, default=1)
    parser.add_argument('--ridge_alpha', type=int, default=100)
    parser.add_argument('--poly_degree', type=int, default=2)

    args = parser.parse_args()

    data_version = args.data_version
    ridge_alpha = args.ridge_alpha
    poly_degree = args.poly_degree

    # read train , eval data
    logging.info(f"Reading train data [version: {data_version}]and val data ")
    train_file_path = f'clean/new_train_{DATA_VERSION}.csv'
    val_file_path = 'split/val.csv'
    
    data_obj = ReaderSaverData()
    train_df = data_obj.read_data(train_file_path)
    val_df = data_obj.read_data(val_file_path)
    
    # feature extraction
    logging.info(f"Starting of feature extraction.")
    train_df, val_df = FeatureExtracting(train_df).apply_feat_extr(), FeatureExtracting(val_df).apply_feat_extr()
    

    logging.info(f"Starting of feature transformation.")
    features = ['is_workday', 'binned_hour', 'log_10_distance', 'x_pickup', 'y_pickup', 'z_pickup', 'x_dropoff','y_dropoff', 'z_dropoff', 'distance_x_y_z']
    trans_train, trans_val, features_name = FeatureTransformtion().apply(train_df[features], val_df[features])

    logging.info(f"Saving the transformed features.")
    transformed_train_df = pd.DataFrame(trans_train, columns=features_name)
    data_obj.save_data(train_df, 'transformed_train_df')


    pipeline = Pipeline(steps=[
        
        ('poly',  PolynomialFeatures(poly_degree, interaction_only=False)),    
        ('Ridge',  Ridge(alpha=ridge_alpha))
    ])

    logging.info(f"fitting the model")
    model = pipeline.fit(trans_train, train_df.log_trip_duration)

    predict_eval(model, trans_train, train_df.log_trip_duration, "train")
    predict_eval(model, trans_val, val_df.log_trip_duration, "val")
