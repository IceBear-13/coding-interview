import pandas as pd
from datetime import datetime
import numpy as np
from scipy.stats import zscore
import math

df = pd.read_csv("Arrest_Data_from_2010_to_2019.csv")

def get_2018():
    val = 0
    for i in df['Arrest Date']:
        year = i[-4:]
        if year == '2018': val+=1
    return val

def get_most_area():
    area_counts = df.loc[df['Arrest Date'].str[-4:] == '2018', 'Area Name'].value_counts()
    max_key = area_counts.idxmax()
    max_value = area_counts[max_key]
    return max_key, max_value

def get_quantile():
    filtered_df = df[(df['Charge Group Description'].isin(['Vehicle Theft', 'Robbery', 'Burglary', 'Receive Stolen Property'])) & (pd.to_datetime(df['Arrest Date'], format='%m/%d/%Y').dt.year == 2018)]
    return filtered_df['Age'].quantile(0.95)

def get_zvalue():
    filtered_df = df[(df['Arrest Date'].str[-4:] == '2018') & ~df['Charge Group Description'].isin(['Pre-Delinquency', 'Non-Criminal Detention']) & df['Charge Group Description'].notna()]

    average_age_per_group = filtered_df.groupby('Charge Group Description')['Age'].mean()
    z_scores = zscore(average_age_per_group)
    largest_absolute_z_score = np.max(np.abs(z_scores))
    return largest_absolute_z_score

def get_distance(phi1, phi2, lamb1, lamb2):
    phi_m = math.radians((phi1+phi2)/2)
    delta_phi = math.radians(phi1-phi2)
    delta_lamb = math.radians(lamb1-lamb2)
    return 6371*math.sqrt(delta_phi**2 +(math.cos(phi_m)*delta_lamb)**2)
        
def get_bradbury():
    lambda2 = 34.050536
    phi2 = -118.247861
    filtered_df = df[(df['Arrest Date'].str[-4:] == '2018') & df['Location']]
    twokmrange = 0

    for index, row in filtered_df.iterrows():
        cords = str(row['Location'])
        cords = cords.replace("(", "").replace(")", "")
        lambda1, phi1 = map(float, cords.split(", "))
        
        delta_lambda = math.radians(lambda1-lambda2)
        delta_phi = math.radians(phi1-phi2)
        phi_m = math.radians((phi1+phi2)/2)
        D = 6371*math.sqrt((delta_phi)**2 + (math.cos(phi_m)*(delta_lambda))**2)
        if D <= 2:
            twokmrange+=1
        
    return twokmrange


def get_pico():
    filtered_df = df.loc[(df['Arrest Date'].str[-4:] == '2018') & df['Address'].str.contains('PICO', case=True)].copy()
    lat = []
    lon = []
    for index, row in filtered_df.iterrows():
        cords = str(row['Location'])
        cords = cords.replace("(", "").replace(")", "")
        lambda1, phi1 = map(float, cords.split(", "))
        lat.append(lambda1)
        lon.append(phi1)
    filtered_df['Latitude'] = lat
    filtered_df['Longitude'] = lon
    lat_std, lon_std = filtered_df['Latitude'].std(), filtered_df['Longitude'].std()
    lat_mean, lon_mean = filtered_df['Latitude'].mean(), filtered_df['Longitude'].mean()
    for index, row in filtered_df.iterrows():
        if row['Latitude'] > lat_mean + 2*lat_std or row['Longitude'] > lon_mean + 2*lon_std:
            filtered_df.drop(index, inplace=True)
    west_lon, east_lon = filtered_df['Longitude'].min(), filtered_df['Longitude'].max()
    west_lat, east_lat = filtered_df.loc[filtered_df['Longitude'] == west_lon]['Latitude'].values[0], filtered_df.loc[filtered_df['Longitude'] == east_lon]['Latitude'].values[0]
    length = get_distance(west_lat, east_lat, west_lon, east_lon)
    return len(filtered_df)/length


def get_probability():
    filtered_df = df[(df['Arrest Date'].str[-4:] < '2019')]
    filtered_df = filtered_df.dropna(subset=['Charge Group Code'])
    filtered_df = filtered_df[filtered_df['Charge Group Code'] != 99]
    paib = filtered_df.groupby(['Area ID', 'Charge Group Code']).size() / filtered_df.groupby('Area ID').size()
    pb = filtered_df.groupby('Charge Group Code').size() / len(filtered_df)
    ratios = paib / pb
    top_5_ratios = ratios.nlargest(5)
    average_ratio = top_5_ratios.mean()
    return average_ratio

    


