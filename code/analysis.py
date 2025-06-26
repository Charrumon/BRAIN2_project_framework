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
    print (maximum_length)
    mean_set = np.zeros(maximum_length)
    std_set = np.zeros(maximum_length)

    #goes through each timestep to find the mean of existing values, assumes common start time
    for index in range(maximum_length):
        running_set = []
        for current_set in multiple_norm:
            if index <= len(current_set):
                running_set.append(multiple_norm[current_set][index])
        print (running_set)        
        mean_set[index] = np.mean(running_set)
        std_set[index] = np.std(running_set)

    return mean_set, std_set



def plot_every_dataset (datasets):
    #takes a dict of file names and data and graphs each

    rows = len(datasets)//2
    columns = rows + (len(datasets) % 2)
    fig, axis = plt.subplots(nrows = rows, ncols = columns)

    plt.axis('off')
    
    titles = title_cleaner (datasets)

    for index in range(len(datasets)):
        column = (index % columns)
        row = index // columns
        data = list(datasets.values())[index]
        axis[row, column].plot(range(len(data)), data)
        axis[row, column].set_title(titles[index])
    plt.subplots_adjust(top= .86, right = .98, wspace = .22, left = .083, hspace = .27)
    plt.suptitle("normalized data of each dataset")
    plt.show()



def plot_mean_dataset (multiple_norm):
    mean_dataset, std_dataset = average_norms(multiple_norm)
    print (mean_dataset)
    x = range(len(mean_dataset))

    plt.plot(x, mean_dataset)
    plt.title("mean of all datasets")

    plt.fill_between(x, mean_dataset - std_dataset, mean_dataset + std_dataset, alpha = .3)
    plt.show()

def title_cleaner (multiple_norm):
    titles = list(multiple_norm)
    
    for i in range(len(titles)):
        titles [i] = titles [i].replace ("d", "day ")
        titles [i] = titles [i].replace ("W", "Wildtype ")
        titles [i] = titles [i].replace ("M", "Mutant ")
        titles [i] = titles [i].replace ("_", " ")
        titles [i] = titles [i].replace (".csv", "")

    return titles

def plot_corr_matrix (multiple_norm):
    all_data = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in multiple_norm.items() ]))
    matrix = all_data.corr()

    #masking data here to get rid of symmetric redundancy
    lower_mask = np.tri(matrix.shape[0], matrix.shape[1], k=-1)
    masked_matrix = np.ma.array(matrix, mask = lower_mask)

    plt.imshow(masked_matrix, cmap ='cividis')
    norm_axes = range(len(multiple_norm))
    labels = title_cleaner (multiple_norm)
    plt.xticks(norm_axes, labels)
    plt.yticks(norm_axes, labels)
    for y in norm_axes:
        for x in norm_axes:
            point_value = masked_matrix[x,y]
            if isinstance(point_value,np.float64):
                plt.text(y,x,round(point_value,6), ha="center", va="center")
        
    plt.subplots_adjust(bottom = .064 , top = .926)
    plt.title("correlation matrix heatmap between given datasets")
    plt.text(-0.2, matrix.shape[0]-1, "R values are \n between 1 and -1", ha="left", bbox={'facecolor': 'white', 'pad': 8})
    plt.show()








hardpath = "C://Users/luisa/Desktop/codeproject/data/tutorial/Data1"

base_datasets = dataloadcsvfolder(hardpath)
normalized_mean_datasets = oddevennormal(base_datasets, 2)



#plots each of the individual datasets
plot_every_dataset(normalized_mean_datasets)


#plots the average dataset, with standard deviation
plot_mean_dataset(normalized_mean_datasets)

#plots the correlation matrix
#plot_corr_matrix(normalized_mean_datasets)



