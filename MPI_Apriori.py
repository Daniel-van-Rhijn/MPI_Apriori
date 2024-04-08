"""
Author:     Daniel van Rhijn
Created:    19/03/2024
Updated:    06/04/2024
Purpose:    Use MPI to execute the A-Priori Algorithm in parallel

Stages:
    1. Initialize all processors to have a section of the data
    2. Each processor gets total counts for it's section. Sends counts back
    to the root node.
    3. Root note identifies frequent terms based on total counts.
    4. Root process sends new round of candidates to each process.
    5. Processes count occurences of candidates and return results to the root process.
    4. Repeat until no new candidates are found to be checked

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
        data - Python 2D List of data items
        categories - Python dictionary with unique items in the data as index
                     names. All values are initialized to 0.
    """
    #Open file to read input from
    f = open(fp, 'r')

    #Define the data structures to return the information in
    categories = {}
    data = []

    #Read lines from the CSV, and add them to the appropriate data structures
    lines = f.readlines()
    for line in lines:
        temp = line.strip().split(sep = ',')
        data.append(temp)
        for item in temp:
            if item not in categories.keys():
                categories[str(item)] = 0
    
    return data, categories

def get_counts_initial(data, counts):
    """
    Purpose:
        Count the amount of times each value is true in a binary dataset.
    Inputs:
        data - 2D Python list containing instances composed of boolean values
    Returns:
        counts - Python dictionary containing integer values representing the number of
                 occurences of each item in the data.
    """
    for basket in data:
        for item in basket:
            counts[str(item)]+=1

    return

def check_frequent(candidates, counts, m, mins):
    """
    Purpose:
        Determine which candidate values occur frequently enough to be
        considered frequent items
    Inputs:
        candidates - A list of candidate association rules to check
        counts - A dictionary of occurences of each candidate in the dataset
        m - Total amount of instances/baskets analyzed
        mins - Minimum support value for a candidate to be considered
    Returns:
        support_frequent - Python dictionary support values of frequent items
        frequent_items - Python list containing lists of frequent itemsets
    """

    #Initialize needed data structures to return
    support_values = {}
    support_frequent = {}
    frequent_items = []

    #Determine support value for each candidate item
    for item_set in candidates:
        support_values[str(item_set)] = float("{:.2f}".format(counts[str(item_set)] / m))
        if support_values[str(item_set)] >= mins:
            frequent_items.append(item_set)
            support_frequent[str(item_set)] = support_values[str(item_set)]
    
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
        support_frequent - Python dictionary support values of frequent items
        frequent_items - Python list containing lists of frequent itemsets
    """

    #Initialize required data structures
    support_values = {}
    support_frequent = {}
    frequent_items = []
    
    #Determine support values for each item
    for item in counts:
        support_values[str([item])] = float("{:.2f}".format(counts[item] / m))
        if support_values[str([item])] >= mins:
            frequent_items.append([item])
            support_frequent[str([item])] = support_values[str([item])]
    
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
        counts - Python dictionary containing number of occurences of each association
                 rule in the dataset
    """

    #Initialize required data structures
    counts = {}
    for item_set in candidates:
        counts[str(item_set)] = 0

    #Check each association rule against every basket/instance
    for basket in data:
        for item_set in candidates:
            if set(item_set).issubset(set(basket)):
                counts[str(item_set)]+=1

    return counts

def get_permutations(base, val_list, max_len):
    """
    Purpose:
        Recursively get all permutations of given list up to the defined size limit
    
    Inputs:
        base - Section of the list to add values to to create new permutations
        list - Values available to add to base
        max_len - Maximum lengths the permutations should reach
    
    Returns:
        permutations - Python list of all permutations up to the size cap
    """
    permutations = []

    for x in range(len(val_list)):
        temp = base + [val_list[x]]
        permutations.append(temp)
        if len(temp) != max_len:
            permutations+=get_permutations(temp, val_list[x+1:], max_len)
    
    return permutations

def parallel_associations(frequent_items, support_values, chunks):
    """
    Purpose:
        Determine the association rules in the frequent_items specified by chunks
    
    Inputs:
        frequent_items - 3D array containing sets of items that are frequent
        support_vales - Python list of dictionaries containing the support values for frequent items
        chunks - 2D python list containing start and stop indexes indicating which
                 part of frequent_items to check.
    
    Outputs:
        associations - Array of associations that have sufficiently high confidence and interest
    """
    associations = [[] for i in range(len(frequent_items) - 1)]


    for k in range(1, len(frequent_items)):
    #For each layer
        for item_set in frequent_items[k][chunks[k][0]:chunks[k][1]]:
            #For each item
            permutations = get_permutations([], item_set, len(item_set)-1)
            for subset in permutations:

                #Determine confidence in the association
                confidence = support_values[k][str(item_set)] / support_values[len(subset)-1][str(subset)]
                interest = confidence - support_values[len(subset)-1][str(subset)]
                if confidence >= 0.50 and abs(interest) >= 0.25:
                    associations[k-1].append((subset, list(set(item_set) - set(subset)), float("%.2f" % confidence), float("%.2f" % interest)))

    return associations

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
        support_values - Python list of dictionaries containing support values for each association
        associations - 2D list of association rules that have high enough confidence and interest
    """

    #Initialize MPI4py variables
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    #Define variables for data division
    start_index = 0
    end_index = 0
    counts = {}
    chunks = []
    
    if rank == 0:
        #Initialize constant values in the root process
        input_file = fp
        support_values = [{}]
        frequent_items = [{}]

        #Import the data and category names into lists
        data, counts = import_data(input_file)
        m = len(data)

        #Divide the data into chunks for distribution
        for i in range(size):
            if (i < len(data)%size):
                end_index = start_index + math.floor(len(data)/size) + 1
            else:
                end_index = start_index + math.floor(len(data)/size)
            chunks.append(data[start_index:end_index])
            start_index = end_index
    
    #Distribute chunks among all processes
    chunks = comm.scatter(chunks, root=0)
    
    #Inform the processes of the counting data structure
    counts = comm.bcast(counts)

    #Get counts for the first layer of items
    get_counts_initial(chunks, counts)

    #Collect each instance of counts back to the root process. Sum all counts together.
    #NOTE: Need to update functionality. Can only reduce properly with numpy arrays. Fix later.
    temp = np.asarray(list(counts.values()))
    temp = comm.reduce(temp, MPI.SUM, root=0)
    if rank == 0:
        temp = temp.tolist()
        i = 0
        for x in counts:
            counts[x] = temp[i]
            i+=1
    else:
        counts = None

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
            temp = np.asarray(list(counts.values()))
            temp = comm.reduce(temp, MPI.SUM, root=0)
            if rank == 0:
                temp = temp.tolist()
                i = 0
                for x in counts:
                    counts[x] = temp[i]
                    i+=1
            else:
                counts = None

            #Determine if any candidates are frequent and add them to the appropriate array
            if rank == 0:
                support_values.append({})
                frequent_items.append({})
                support_values[k], frequent_items[k] = check_frequent(candidates, counts, m, mins)
        
        #Advance K by 1, exit if no candidates were provided this round
        k+=1
        if k == 100 or (len(candidates) == 0):
            termination = True
    
    #Determine chunks of the frequent_items to be passed to each process
    if rank == 0:
        chunks = [[] for i in range(size)]
        for layer in frequent_items:
            end_index = 0
            start_index = 0
            for i in range(size):
                if (i < len(layer)%size):
                    end_index = start_index + math.floor(len(layer)/size) + 1
                else:
                    end_index = start_index + math.floor(len(layer)/size)
                
                chunks[i].append((start_index, end_index))
                start_index = end_index
    
    #Distribute the required information for finding the associations
    if rank != 0:
        frequent_items = []
        support_values = []
    chunks = comm.scatter(chunks, root=0)
    frequent_items = comm.bcast(frequent_items, root=0)
    support_values = comm.bcast(support_values, root=0)
    
    associations = parallel_associations(frequent_items, support_values, chunks)

    #Gather the computed associations
    temp = []
    temp = comm.gather(associations, root = 0)
    if rank == 0:
        associations = []
        for i in range(len(temp[0])):
            associations.append([])
            for j in range(len(temp)):
                associations[i]+=temp[j][i]

    if rank == 0:
        return frequent_items, support_values, associations
    
    return