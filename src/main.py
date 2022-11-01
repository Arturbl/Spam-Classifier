import pandas as pd
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


THRESHOLD = 100
STOP_WORDS = stopwords.words("english")  # get regular words that can be ignored for example (it, from, to, how)
TOKEN_COUNTER = {}


# get dataframe
def get_data_frame():
    csv_path = "utils/spam.csv"
    return pd.read_csv(csv_path, encoding='ISO-8859-1')


# transform a message/email into a list of tokens
# tokenizer:
# test_message = "Hey,,, GGggGG feet it going? <HTML><bads> bads 'randoms' badly"
# test_message_tokenized = tokenizer.tokenize(test_message)
# test_message_tokenized = ['Hey','GGggGG','feet','it','going','HTML','bads','bads','randoms','badly']
def message_to_token_list(message):
    tokenizer = nltk.RegexpTokenizer(r"\w+") # A tokenizer that splits a string using a regular expression
    tokens = tokenizer.tokenize(message)
    lowercased_tokens = [t.lower() for t in tokens]
    lemmatized_tokens = [WordNetLemmatizer().lemmatize(t) for t in lowercased_tokens]
    useful_tokens = [t for t in lemmatized_tokens if t not in STOP_WORDS]
    return useful_tokens


# check how many times each token appears in the csv file and store them into a map
def generate_vocabulary(train_df):
    for message in train_df["v2"]:
        message_as_token_lst = message_to_token_list(message)
        for token in message_as_token_lst:
            if token in TOKEN_COUNTER:
                TOKEN_COUNTER[token] += 1
            else:
                TOKEN_COUNTER[token] = 1


# determines which tokens should be considered according to the amount of times it appeared
# keep_token("euro", 10) -> return false because the token "euro" does not appear more than 10 times
# processed_token -> each token in token_counter
# threshold > positive integer
def keep_token(processed_token):
    if processed_token not in TOKEN_COUNTER:
        return False
    else:
        return TOKEN_COUNTER[processed_token] > THRESHOLD


# create vector and count appearences of tokens in a single message/email
def message_to_count_vector(message):
    count_vector = np.zeros(len(bag_of_words))
    processed_list_of_tokens = message_to_token_list(message)
    for token in processed_list_of_tokens:
        if token not in bag_of_words:
            continue
        index = token_to_index_mapping[token]
        count_vector[index] += 1
    return count_vector


def generate_results(data_frame):
    y = [i for i in data_frame['v1']]  # [0 if i == 'ham' else 1 for i in data_frame['v1']]
    message_col = data_frame["v2"]
    count_vectors = []
    for message in message_col:
        count_vector = message_to_count_vector(message)
        count_vectors.append(count_vector)
    x = np.array(count_vectors).astype(int)
    return x, y


# UNCOMMENT THIS IF IT IS THE FIRST RUN
# nltk.download("wordnet")
# nltk.download("stopwords")
# nltk.download('omw-1.4')
df = get_data_frame()
split_index = int(len(df) * 0.7)  # the first 70% is for train and the last 30 is for test
train_df, test_df = df[:split_index], df[split_index:]


generate_vocabulary(train_df)

bag_of_words = set()  # bag of words, we use sets because sets cant have duplicates
for token in TOKEN_COUNTER:
    if keep_token(token):
        bag_of_words.add(token)
bag_of_words = list(bag_of_words)

# create a map with the bag of words and give an index to each one
token_to_index_mapping = {t: i for t, i in zip(bag_of_words, range(len(bag_of_words)))}

x_train, y_train = generate_results(train_df)
x_test, y_test = generate_results(test_df)

lr = LogisticRegression().fit(x_train, y_train)
print(classification_report(y_test, lr.predict(x_test)))
