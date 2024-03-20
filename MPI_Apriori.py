"""
Author:     Daniel van Rhijn
Created:    19/03/2024
Updated:    20/03/2024
Purpose:    Use MPI to execute the A-Priori Algorithm in parallel

Stages:
    1. Initialize all processors to have a section of the data
    2. Each processor gets total counts for it's section. Sends counts back
    to the root node.
    3. Root note identifies frequent terms based on total counts.
    4. Root process sends new round of candidates to each process.
    5. Processes count occurences of candidates and return results to the root process.
    4. Repeat until no new candidates are found to be checked

Improvements:
    1. Use numpy arrays throughout the whole algorithm rather than lists. Saves converting back and forth constantly
    2. Consider speedup by doing redundant calculations on each node rather than communicating more(Implement allreduce, etc.)

CRITICAL NOTES:
    1. There is a hard coded limit for association sizes to 100. If sizes larger than this are expected adjust the limiter. Consider
    adding a feature that can disable it without adjusting the code.
"""

from mpi4py import MPI
import pandas as pd
import numpy as np
import math

def import_data(fp):
    """
    Purpose:
        Import data from a CSV into python lists
    Inputs:
        fp - Path of a CSV file containing the dataset to be imported
    Returns:
        data - Python 2D List of 
    """

    df = pd.read_csv(fp)
    data = df.values.tolist()
    categories = df.columns.tolist()

    return data, categories

def get_counts_initial(data):
    """
    Purpose:
        Count the amount of times each value is true in a binary dataset.
    Inputs:
        data - 2D Python list containing instances composed of boolean values
    Returns:
        counts - Python List containing integer values representing the number of
                 positive occurences of each feature in the data.
    """
    counts = [0] * len(data[0])
    for basket in data:
        for n in range(len(basket)):
            if basket[n] == 1:
                counts[n]+=1

    return counts

def check_frequent(candidates, counts, m, mins):
    """
    Purpose:
        Determine which candidate values occur frequently enough to be
        considered frequent items
    Inputs:
        candidates - A list of candidate association rules to check
        counts - A list of occurences of each candidate in the dataset
        m - Total amount of instances/baskets analyzed
        mins - Minimum support value for a candidate to be considered
    Returns:
        support_values - Python list of support values for each candidate
        support_frequent - Python list containing support values of frequent itemsss
    """

    #Initialize needed data structures to return
    support_values = [0] * len(counts)
    support_frequent = []
    frequent_items = []

    #Determine support value for each candidate item
    for n in range(len(support_values)):
        support_values[n] = float("{:.2f}".format(counts[n] / m))
    
    #Add the candidates with high enough support values to the frequent_items array
    for n in range(len(support_values)):
        if support_values[n] >= mins:
            frequent_items.append(candidates[n])
            support_frequent.append(support_values[n])
    
    return support_frequent, frequent_items

def generate_candidates(frequent_items, k):
    """
    Purpose:
        Generate possible combinations of frequent items that have
        a size of k
    Inputs:
        frequent_items - 2D Python List of frequent associations of size k-1
        k - Size of candidate groups to generate
    Returns:
        candidates - 2D Python List containing possible pairings of frequent items 
                     such that each one new item is added to each association.
    """

    item_count = len(frequent_items)
    candidates = []
    temp = []

    #Consider each item relative to the others
    for n in range(item_count):
        for j in range(n, item_count):
            #Create a set that contains all items in both groups
            temp = list(set(frequent_items[n]).union(set(frequent_items[j])))

            #Add the new candidate if it's the right size, and if it hasn't already been added
            if len(temp) == k+1 and temp not in candidates:
                #Check all subsets are in previous layer
                acceptable = True
                for q in range(k+1):
                    if (temp[0:q] + temp[q+1:k+1]) not in frequent_items:
                        acceptable = False
                if acceptable:
                    candidates.append(temp)

    return candidates

def generate_candidates_initial(counts, m, mins):
    """
    Purpose:
        Generate candidate groups of size 1 based off of how often
        they occur in the dataset
    Inputs:
        counts - A list of positive occurences of each feature in the dataset
        m - Total amount of instances/baskets analyzed
        mins - Minimum support value for a candidate to be considered
    Returns:
        data - Python 2D List of 
    """

    #Initialize required data structures
    support_values = [0] * len(counts)
    support_frequent = []
    frequent_items = []
    
    #Determine support values for each item
    for n in range(len(support_values)):
        support_values[n] = float("{:.2f}".format(counts[n] / m))
    
    #Add items with a high enough support value to the frequent_items list
    for n in range(len(support_values)):
        if support_values[n] >= mins:
            frequent_items.append([n])
            support_frequent.append(support_values[n])
    
    return support_frequent, frequent_items

def get_counts(candidates, data):
    """
    Purpose:
        Compute how many times sets of features are true simultaniously in a
        dataset
    Inputs:
        candidates - 2D Python List of candidate association rules to count the
                     occurences of in the dataset
        data - 2D Python List containing boolean data
    Returns:
        counts - Python list containing number of occurences of each association
                 rule in the dataset
    """

    #Initialize required data structures
    counts = [0] * len(candidates)

    #Check each association rule against every basket/instance
    for n in range(len(data)):
        for j in range(len(candidates)):
            contains = True
            for x in candidates[j]:
                if data[n][x] != 1:
                    contains = False
            if contains == True:
                counts[j]+=1
    
    return counts

def apriori(fp, mins):
    """
    Purpose:
        Use MPI4Py to execute the A-Priori algorithm on a given dataset
    Inputs:
        fp - File pointer to a dataset of boolean values to execute the algorithm on
        mins - Minimum support value for an item in the dataset to be considered frequent.
               Intended as a float value between 1 and 0.
    Returns:
        frequent_items - 3D Python List containing associations rules of various sizes
        support_values - 2D Python List containing support values for each association
        categories - List of feature names that make up the dataset
    """

    #Initialize MPI4py variables
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()


    #Define variables for data division
    start_index = 0
    end_index = 0
    chunks = []
    

    if rank == 0:
        #Initialize constant values in the root process
        input_file = fp
        support_values = [[]]
        frequent_items = [[]]


        #Import the data and category names into lists
        data, categories = import_data(input_file)
        m = len(data)


        #Divide the data into chunks for distribution
        for i in range(size):
            if (rank < len(data)%size):
                end_index = start_index + math.floor(len(data)/size) + 1
            else:
                end_index = start_index + math.floor(len(data)/size)
            chunks.append(data[start_index:end_index])
            start_index = end_index
        

    #Distribute chunks among all processes
    chunks = comm.scatter(chunks, root=0)


    #Get counts for the first layer of items
    counts = get_counts_initial(chunks)

    #Collect each instance of counts back to the root process. Sum all counts together.
    #NOTE: Need to update functionality. Can only reduce properly with numpy arrays. Fix later.
    counts = np.asarray(counts)
    counts = comm.reduce(counts, MPI.SUM, root=0)
    if rank == 0:
        counts = counts.tolist()
    else:
        counts = []

    #Determine the first layer of frequent objects to consider
    if rank == 0:
        support_values[0], frequent_items[0] = generate_candidates_initial(counts, m, mins)
    
    #Loop until the root process sends a list with 0 candidates
    termination = False
    k = 1
    while not termination:
        #Initialize and share the list of candidates for each process to consider
        if rank == 0:
            candidates = generate_candidates(frequent_items[k-1], k)
        else:
            candidates = []
        
        #Distribute new round of candidate associations to consider
        candidates = comm.bcast(candidates, root=0)

        #If candidates exist for this size then determine associated counts
        if len(candidates) != 0:
            counts = get_counts(candidates, chunks)

            #Sum all counts and return them to the root processs
            counts = np.asarray(counts)
            counts = comm.reduce(counts, MPI.SUM, root=0)
            if rank == 0:
                counts = counts.tolist()
            else:
                counts = []

            #Determine if any candidates are frequent and add them to the appropriate array
            if rank == 0:
                support_values.append([])
                frequent_items.append([])
                support_values[k], frequent_items[k] = check_frequent(candidates, counts, m, mins)
        
        #Advance K by 1, exit if no candidates were provided this round
        k+=1
        if k == 100 or (len(candidates) == 0):
            termination = True

    if rank == 0:
        return frequent_items, support_values, categories
    
    return
    
    
    