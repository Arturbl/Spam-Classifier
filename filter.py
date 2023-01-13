import csv
import subprocess
from naive_bayes import *
from percetron import *


def main():
    output=subprocess.check_output(['wc','-l','spam.csv'])
    num_lines=int(output.decode().split()[0])

    training_set = list() 
    test_set = list() 
    validation_set = list()


    with open("spam.csv", "r") as spam_file:
        file = list(csv.reader(spam_file))
        for i in range(num_lines-2):
            message= file[i]
            if i < round(num_lines*0.7):
                training_set.append(message)
            elif i < round(num_lines*0.85):
                test_set.append(message)
            else:
                validation_set.append(message)


#################################NAIVE BAYES################################    
    # Train the classifier on the training set
    bayes_training(training_set)
    
    # Adjust the value of c to optimize the classifier's performance
    adjust_c(validation_set)
    
    # Test the classifier on the test set
    bayes_test(test_set)
    
    # Initialize the statistics
    stats = {"guesses":0,"c_guesses":0,"w_guesses":0,"spam":0,"ham":0,"t_pos":0,"t_neg":0,"f_pos":0,"f_neg":0}
    
    # Return the statistics for the classifier's performance on the test set
    stats_updated= return_stats(stats, test_set)
    print("## Naive Bayes ##")
    metrics(stats_updated)

###################################PERCETRON##################################  

    teta,zero = perceptron(training_set,15)

    percetron_stats = {"guesses":0,"c_guesses":0,"w_guesses":0,"spam":0,"ham":0,"t_pos":0,"t_neg":0,"f_pos":0,"f_neg":0}

    percetron_with_metrics(validation_set, teta,zero, percetron_stats)

    print()
    print("## Percetron ##")
    metrics(percetron_stats)


    
def metrics(stats_dic):
    print("Recall: ", (stats_dic["t_pos"] / (stats_dic["t_pos"] + stats_dic["t_neg"])))
    print("Specificity: ", (stats_dic["t_neg"] / (stats_dic["t_neg"] + stats_dic["f_pos"])))
    print("Precision: ", (stats_dic["t_pos"] / (stats_dic["t_pos"] + stats_dic["f_pos"])))
    print("Sensivity: ", (stats_dic["t_pos"] / (stats_dic["t_pos"] + stats_dic["f_neg"]))) 
    print("Accuracy: ", (stats_dic["c_guesses"] / stats_dic["guesses"]))
    print("Error Rate: ", (stats_dic["w_guesses"] / stats_dic["guesses"]))

if __name__=='__main__':
    main()
