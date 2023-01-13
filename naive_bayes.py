
import math
import string
spam_words = {}
ham_words = {}
spam_count = [0,0]
ham_count = [0,0]
c= 0.001

def preprocess_text(text):
    # Lowercase and remove punctuation
    words = text.lower().translate(str.maketrans('', '', string.punctuation)).split(" ")
    
    return words

def calculate_offset():
   
    return -(math.log(c) + math.log(ham_count[1]) - math.log(spam_count[1]))

def classify_text(text):
    offset = calculate_offset()
    
    prepd_mail = text.lower().split(" ")
    
    for word in prepd_mail:
        new_word = word.translate(str.maketrans('', '', string.punctuation))
        spam_entry = spam_words.get(new_word, [0, 0])
        ham_entry = ham_words.get(new_word, [0, 0])
        spam_freq = spam_entry[0]
        ham_freq = ham_entry[0]
        if spam_freq > 0 and ham_freq > 0:
            offset += (math.log(spam_freq) - math.log(ham_freq))
    
    if offset > 0:
        return "spam"
    return "ham"


def adjust_c(data_list):
    global c
    prev_validation = 0
    # Iterate over the data 15 times
    for _ in range(15):
        c_guesses = 0
        # Classify each data point and count the number of correct guesses
        for data_point in data_list:
            bayes = classify_text(data_point[1])
            if bayes == data_point[0]:
                c_guesses += 1
        prev_validation = (c_guesses/len(data_list))
        
    # Iterate over the data 15 more times
    for _ in range(15):
        c_guesses = 0
        for data_point in data_list:
            bayes = classify_text(data_point[1])
            if bayes == data_point[0]:
                c_guesses += 1
        validation = (c_guesses/len(data_list))
        
        # If the classifier's performance worsens, break out of the loop
        if prev_validation > validation:
            break
        else:
            prev_validation = validation
            c *= 2.5
    


def bayes_training(data_list):
    # Iterate over the data
    for data_point in data_list:
        # If the label is "spam", update the spam count and frequencies
        if data_point[0] == "spam":
            spam_count[1] += 1
            prepd_data = data_point[1].lower().split(" ")
            for word in prepd_data:
                new_word = word.translate(
                    str.maketrans('', '', string.punctuation))
                if new_word.lower() not in spam_words:
                    spam_words[new_word] = [1, 0]
                else:
                    spam_words[new_word][0] += 1
                spam_count[0] += 1
        
        # If the label is "ham", update the ham count and frequencies
        elif data_point[0] == "ham":
            ham_count[1] += 1
            prepd_data = data_point[1].lower().split(" ")
            for word in prepd_data:
                new_word = word.translate(
                    str.maketrans('', '', string.punctuation))
                if new_word.lower() not in ham_words:
                    ham_words[new_word] = [1, 0]
                else:
                    ham_words[new_word][0] += 1
                ham_count[0] += 1
    
    # Calculate the relative frequencies of the words in spam and ham messages
    for word in spam_words:
        spam_words[word][1] = spam_words[word][0] / spam_count[0]
    for word in ham_words:
        ham_words[word][1] = ham_words[word][0] / ham_count[0]

def bayes_test(data_list):
    c_guesses = 0
    w_guesses = 0
    
    # Iterate over the data
    for data_point in data_list:
        # Classify the text
        bayes = classify_text(data_point[1])
        
        # Update the statistics based on the classification
        if bayes == data_point[0]:
            c_guesses +=1
        else:
            w_guesses += 1

    return (c_guesses, w_guesses)

def return_stats(stats, data_list):
    # Iterate over the data
    for data_point in data_list:
        stats["guesses"] += 1
        if data_point[0] == "spam":
            stats["spam"] += 1
        else:
            stats["ham"] += 1
        
        # Classify the text
        bayes = classify_text(data_point[1])
        
        # Update the statistics based on the classification
        if bayes == "spam" and data_point[0] == "spam":
            stats["t_pos"] += 1
        elif bayes == "ham" and data_point[0] == "ham":
            stats["t_neg"] += 1
        elif bayes == "spam" and data_point[0] == "ham":
            stats["f_pos"] += 1
        elif bayes == "ham" and data_point[0] == "spam":
            stats["f_neg"] += 1
    
    # Calculate the number of correct and incorrect classifications
    stats["c_guesses"] = stats["t_pos"] + stats["t_neg"]
    stats["w_guesses"] = stats["f_pos"] + stats["f_neg"]
    
    return stats