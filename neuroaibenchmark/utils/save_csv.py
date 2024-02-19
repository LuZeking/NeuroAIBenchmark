import os
import csv

def save_csv(results, path='results.csv'):
    '''Save the results of the benchmarking to a csv file.'''

    # Create the csv file
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file)

        # Write the header, r.g. ['Model', 'Dataset', 'Metric', 'Value']
        writer.writerow(results) 

        # Write the data
        for result in results:
            writer.writerow(result)

def print_results(results):
    '''Print the results of the benchmarking to the console.'''
    for result in results:
        print(result)