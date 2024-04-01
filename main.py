"""
Author: Daniel van Rhijn
Date: 20/03/2024
Updated: 30/03/2024
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
input_path = "/home/weirdsquid/School Files/CP431-Workspace/Term Project/Data Preprocessing/output.csv"
output_path = "/home/weirdsquid/School Files/CP431-Workspace/Term Project/Version 1.1/output.txt"
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
    frequent_items, support_values, categories, associations = temp

    if output_path == "":
        #Output information about frequent itemsets
        # for n in range(len(frequent_items)):
        #     print("Layer %d Frequent Items:" % (n+1))
        #     for j in range(len(frequent_items[n])):
        #         temp = frequent_items[n][j]
        #         for q in range(len(temp)):
        #             temp[q] = categories[temp[q]]
        #         print(temp, end= " ")
        #         print("Support Value: %.2f" % support_values[n][j])
        #     print("\n")

        #Output information about interesting associations
        for layer in associations:
            for rule in layer:
                for q in range(len(rule[0])):
                    rule[0][q] = categories[rule[0][q]]
                for p in range(len(rule[1])):
                    rule[1][p] = categories[rule[1][p]]
                print(str(rule[0]) + " --> " + str(rule[1]) + " | Confidence: %.2f | Interest: %.2f" % (rule[2], rule[3]))
            print()
    else:
        #Open file to output to
        f = open(output_path, "w")

        #Output information about frequent itemsets
        # for n in range(len(frequent_items)):
        #     f.write("Layer %d Frequent Items:" % (n+1))
        #     for j in range(len(frequent_items[n])):
        #         temp = frequent_items[n][j]
        #         for q in range(len(temp)):
        #             temp[q] = categories[temp[q]]
        #         f.write(str(temp) + " Support Value: %.2f\n" % support_values[n][j])
        #     f.write("\n")
        
        #Output information about interesting associations
        for layer in associations:
            for rule in layer:
                for q in range(len(rule[0])):
                    rule[0][q] = categories[rule[0][q]]
                for p in range(len(rule[1])):
                    rule[1][p] = categories[rule[1][p]]
                f.write(str(rule[0]) + " --> " + str(rule[1]) + " | Confidence: %.2f | Interest: %.2f\n" % (rule[2], rule[3]))
            f.write("\n")
        f.close()
