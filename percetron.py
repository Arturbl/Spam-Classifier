import collections
from collections import Counter
import math

def perceptron(data_set, times):
    data_set = map_labels(data_set)
    data_set = count_word_freq(data_set)

    teta, teta_zero = teta_generator(data_set)
    for _ in range(times):
        for label, words in data_set:
            email_dic = collections.Counter(words)
            if label * percetron_classifier(teta, email_dic, teta_zero) <= 0:
                for word, count in email_dic.items():
                    teta[word] += label * count
                teta_zero += label
    return teta, teta_zero

def percetron_classifier(teta, email_dic, teta_zero):
    total_sum = sum(count * teta.get(word, 0) for word, count in email_dic.items())
    return total_sum + teta_zero 

def map_labels(email_list):
    labels = {'spam': -1, 'ham': 1}
    return [[labels[email[0]], email[1]] for email in email_list]

def count_word_freq(email_list):
    return [[email[0], collections.Counter(email[1].split())] for email in email_list]

def teta_generator(data_set):
    words = {word for email in data_set for word in email[1]}
    teta = dict.fromkeys(words, 0)
    return teta, 0


def percetron_with_metrics(validation_list, teta, teta_zero, stats):
    validation_list = map_labels(validation_list)
    validation_list = count_word_freq(validation_list)

    counter = Counter()
    results = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
    for email in validation_list:
        cleaned_email_dic = {clean_up_word(word): count for word, count in email[1].items()}
        classifier = percetron_classifier(teta, cleaned_email_dic, teta_zero)
        classifier_sign = math.copysign(1, classifier)
        if classifier_sign == email[0]:
            counter['correct'] += 1
            if classifier_sign == -1:
                results['tp'] += 1
            elif classifier_sign == 1:
                results['tn'] += 1
        else:
            counter['incorrect'] += 1
            if classifier_sign == -1:
                results['fp'] += 1
            elif classifier_sign == 1:
                results['fn'] += 1
    stats["guesses"] = counter['correct'] + counter['incorrect']
    stats["spam"] = counter[-1]
    stats["ham"] = counter[1]
    stats["t_pos"]= results['tp']
    stats["t_neg"] = results['tn']
    stats["f_pos"] = results['fp']
    stats["f_neg"] = results['fn']
    stats["c_guesses"] =counter['correct']
    stats["w_guesses"] =counter['incorrect']

def clean_up_word(word):
    for ch in "!?,.-/;_{}%:()<>\\":
        word = word.replace(ch, "")
    return word.lower()