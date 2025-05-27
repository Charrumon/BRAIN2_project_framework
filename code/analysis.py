import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os 
import csv



def dataloadcsvfolder(path):
    # passes an array of csv files held in the given folder directory
    # returns a dict with each file name associated with its data

    folder_dir = os.listdir(path)
    data_arr = {}

    for file_name in folder_dir:
        specific_path = path + "/" + file_name
        if specific_path[-4:] == ".csv" :
            data_arr[file_name] = pd.read_csv(specific_path)

    return data_arr 



def oddevennormal(multiple_csv, range_csv):
    #TO DO add functionality for ranges other than 2
    mean_sets = {}
    
    for csv_data in multiple_csv:
        #TO DO print out exception if non zero
        #exception = data_length % range_csv

        data_length = len(multiple_csv[csv_data])
        available_set_length = data_length//range_csv
        new_mean_set = [0]*available_set_length


        #finds each value of R[odd/even], compares to all values to find R minimum 
        R_minimum = float('inf')

        for index in range(available_set_length):
            zero_index = index * range_csv
            current_ratio = multiple_csv[csv_data].iloc[zero_index,2] / multiple_csv[csv_data].iloc[zero_index+1,2]
            new_mean_set [index] = current_ratio

            #sets running minimum 
            if current_ratio < R_minimum:
                R_minimum = current_ratio

        #Divides each value of the dataset of R[odd/even] values by the minimum odd/even ratio
        new_mean_set = new_mean_set / R_minimum

        #puts the now normalized dataset into the dictionary under its present name
        mean_sets [csv_data] = new_mean_set -1

    return mean_sets



def average_norms (multiple_norm):
    #produces a dataset averaging each point of the given datasets, and 
    # a dataset of standard deviations for each point. Takes a dict

    maximum_length = 0

    #finds the length of the longest dataset
    for current_set in multiple_norm:
        length = len(multiple_norm[current_set])
        if length > maximum_length:
            maximum_length = length

    mean_set = np.zeros(maximum_length)
    std_set = np.zeros(maximum_length)

    #goes through each timestep to find the mean of existing values, assumes common start time
    for index in range(maximum_length):
        running_set = []
        for current_set in multiple_norm:
            running_set.append(multiple_norm[current_set][index])
        mean_set[index] = np.mean(running_set)
        std_set[index] = np.std(running_set)
    return mean_set, std_set



def plot_every_dataset (datasets):
    #takes a dict of file names and data and graphs each

    rows = len(datasets)//2
    columns = rows + (len(datasets) % 2)
    fig, axis = plt.subplots(nrows = rows, ncols = columns)

    plt.axis('off')

    for index in range(len(datasets)):
        column = (index % columns)
        row = index // columns
        data = list(datasets.values())[index]
        axis[row, column].plot(range(len(data)), data)
        axis[row, column].set_title(list(datasets)[index])
    plt.subplots_adjust(top= .95, right = .98, wspace = .22, left = .083, hspace = .27)
    plt.show()



def plot_mean_dataset (multiple_norm):
    mean_dataset, std_dataset = average_norms(multiple_norm)
    x = range(len(mean_dataset))

    plt.plot(x, mean_dataset)
    plt.title("mean of all datasets")

    plt.fill_between(x, mean_dataset - std_dataset, mean_dataset + std_dataset, alpha = .3)
    plt.show()



def plot_corr_matrix (multiple_norm):
    all_data = []
    for point in multiple_norm:
        all_data.append(multiple_norm[point])
    matrix = np.corrcoef(all_data)
    plt.imshow(matrix, cmap ='hot')
    plt.xticks(range(len(multiple_norm)), list(multiple_norm))
    plt.yticks(range(len(multiple_norm)), list(multiple_norm))
    plt.subplots_adjust(bottom = .064 , top = .926)
    plt.title("correlation matrix heatmap between given datasets")
    plt.show()








hardpath = "C://Users/luisa/Desktop/codeproject/data/tutorial/Data1"

base_datasets = dataloadcsvfolder(hardpath)
normalized_mean_datasets = oddevennormal(base_datasets, 2)



#plots each of the individual datasets
#plot_every_dataset(normalized_mean_datasets)


#plots the average dataset, with standard deviation
#plot_mean_dataset(normalized_mean_datasets)

#plots the correlation matrix
plot_corr_matrix(normalized_mean_datasets)



