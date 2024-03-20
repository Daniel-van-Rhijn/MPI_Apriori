"""
Author: Daniel van Rhijn
Date: 20/03/2024
Purpose: Output the information gained from the A-Priori Algorithm into a file with understandable formatting

Instructions for use:
    To execute using command line arguments use formatting:
        mpiexec -n [# of cores] python main.py "input_file_path" minimum_support
    
    Example:
        mpiexec -n 2 python main.py "/home/weirdsquid/School Files/CP431-Workspace/Term Project/Data Preprocessing/output.csv" 0.25

    To execute without using command line arguments:
        Set variables defined at the top of the program to the required values
    
    Example:
        mpiexec -n 2 python main.py
"""

from MPI_Apriori import apriori
import sys

#Define input_path and mins here, or use command line arguments
input_path = ""
output_path = ""
mins = 0.25

#Get command line inputs
args = sys.argv

#Call required function with whichever method was used
if len(args) > 1:
    temp = apriori(args[1], float(args[2]))
else:
    temp = apriori(input_path, mins)

#Output the data in a nice format
if temp != None:
    frequent_items, support_values, categories = temp

    if output_path == "":
        for n in range(len(frequent_items)):
            print("Layer %d Frequent Items:" % (n+1))
            for j in range(len(frequent_items[n])):
                temp = frequent_items[n][j]
                for q in range(len(temp)):
                    temp[q] = categories[temp[q]]
                print(temp, end= " ")
                print("Support Value: %.2f" % support_values[n][j])
            print("\n")
    else:
        f = open(output_path, "w")
        for n in range(len(frequent_items)):
            f.write("Layer %d Frequent Items:" % (n+1))
            for j in range(len(frequent_items[n])):
                temp = frequent_items[n][j]
                for q in range(len(temp)):
                    temp[q] = categories[temp[q]]
                f.write(str(temp) + " Support Value: %.2f\n" % support_values[n][j])
            f.write("\n")
