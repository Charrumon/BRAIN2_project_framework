import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os 
import csv

def dataloadcsv(path):
    # passes an array of csv files held in the given folder directory

    folder_dir = os.listdir(path)
    data_arr = []

    for file_name in folder_dir:
        specific_path = path + "/" + file_name
        if specific_path[-4:] == ".csv" :
           data_arr.append(open(specific_path))
    
    return data_arr 

def oddevennormal(multiple_csv):
    # performs normalization on the given data by splitting it 
    # into odd and even groups, then dividing a ratio of the mean 
    # of those groups by the minimum of all data at the given
    # point in time. Returns an array containing the results of 
    # this normalization.
    normal_data = []

    # splits the data sets into odd and even groups, simultaneously 
    # finds maximum length of dataset, assuming at data starts at 
    # the same point in time
    odd_data = []
    even_data = []
    max_index = float('inf')

    for index in range(len(multiple_csv)):
        current_set = pd.read_csv(multiple_csv[index])

        if index % 2 == 0:
            odd_data.append(current_set)
        else:
            even_data.append(current_set)

        current_max = len(current_set)
        if current_max < max_index:
            max_index = current_max



    # in the case that there are an unequal amount of odd/even datasets, 
    # produces a ratio to balance out the ratio of the means produced 
    # by dividing the sum of all odd means by the sum of all even means
    ratio = 1
    if len(odd_data) > len(even_data):
        ratio = len(even_data) / len(odd_data)

    # gathers and compares data from each timestep until the maximum 
    # possible timestep; std of means, ratio of means sum, and minimum
    # of the minimums

    odd_sum_arr = [0]*max_index
    even_sum_arr = [0]*max_index
    normalization_factor_arr = [0]*max_index
    normalized_df = pd.DataFrame(np.zeros((max_index, len(multiple_csv))))
    for time in range(max_index):
        current_min_arr = []
        mean_arr = np.array([])
        minimum = float('inf')
        for data in odd_data:
            current_mean = data.iloc[time,2]
            odd_sum_arr[time] = odd_sum_arr[time] + current_mean
            mean_arr = np.append(mean_arr, current_mean)
            current_min_arr.append(data.iloc[time,3])
        
        for data in even_data:
            current_mean = data.iloc[time,2]
            even_sum_arr[time] = even_sum_arr[time] + current_mean
            mean_arr = np.append(mean_arr, current_mean)
            current_min_arr.append(data.iloc[time,3])


        for val in current_min_arr:
            if minimum > val:
                minimum = val

        normalization_factor = (((odd_sum_arr[time]*ratio))/even_sum_arr[time])/minimum
        normalized_df.loc[time] = mean_arr * float(normalization_factor)

    return normalized_df






        

        


        
        






        
    

hardpath = "C://Users/luisa/Desktop/codeproject/data/tutorial/Data1"

dataset = dataloadcsv(hardpath)
normalized_dataset = oddevennormal(dataset)

normalized_dataset.to_csv('C://Users/luisa/Desktop/codeproject/data/tutorial/normalized_dataset.csv')



