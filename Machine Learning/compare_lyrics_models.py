import pickle
import numpy as np 
import matplotlib.pyplot as plt  


def provide_labels(methods,values):
    for i in range(len(methods)):
        plt.text(i,values[i],values[i])


def my_function():

    # creating the dataset 
    data = {'RF':70.3,'SVM':64.7,'NB':56.0,'LR':66.3} 
    methods = list(data.keys()) 
    values = list(data.values()) 
    
    fig = plt.figure(figsize = (10, 5)) 
    
    # creating the bar plot 
    plt.bar(methods, values, color ='magenta', width = 0.4)

    provide_labels(methods,values)
    
    plt.xlabel("Methods") 
    plt.ylabel("Accuracy") 
    plt.title("Performance comparison")
    plt.savefig('Project_Extra/l_comparison.png')
    plt.show() 

if __name__=="__main__":
    my_function()