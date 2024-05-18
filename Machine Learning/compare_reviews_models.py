import pickle
import numpy as np 
import matplotlib.pyplot as plt  


def provide_labels(methods,values):
    for i in range(len(methods)):
        plt.text(i,values[i],values[i])


def my_function():

    # creating the dataset 
    data = {'RF':90.9,'SVM':94.6,'NB':74.6,'LR':93.8} 
    methods = list(data.keys()) 
    values = list(data.values()) 
    
    fig = plt.figure(figsize = (10, 5)) 
    
    # creating the bar plot 
    plt.bar(methods, values, color ='green', width = 0.4)

    provide_labels(methods,values)
    
    plt.xlabel("Methods") 
    plt.ylabel("Accuracy") 
    plt.title("Performance comparison")
    plt.savefig('Project_Extra/r_comparison.png')
    plt.show() 

if __name__=="__main__":
    my_function()