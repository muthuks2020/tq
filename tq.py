# Import the required libraries
import csv
from sklearn.model_selection import train_test_split
import re
import pandas as pd

# Read csv and convert to txt
csv_file = 'Dataset/dataset_kyc_non_indian_3011.csv'
txt_file = 'Dataset/dataset_kyc_non_indian_3011.txt'
with open(txt_file, "w") as my_output_file:
    with open(csv_file, "r") as my_input_file:
        [my_output_file.write(" ".join(row)) for row in csv.reader(my_input_file)]
    my_output_file.close()

# preprocess the text and save it as train and test data files
labels = []
text = []
f = open('Dataset/dataset_kyc_non_indian_3011.txt')
lines = f.readlines()
lines.pop(0)
for line in lines:
    text_string = line.split(' ', 1)[1]
    label_string = line.split(' ', 1)[0]
    # Preprocessing
    text.append(re.sub('\W+', ' ', str(text_string.encode('ascii', 'ignore'), 'utf-8')).lower())
    labels.append(label_string)

# split train and test data
train_text, test_text, train_label, test_label = train_test_split(text, labels, stratify=labels, test_size=0.25)

# join labels and data
train_data = {'Labels': train_label, 'data': train_text}
test_data = {'Labels': test_label, 'data': test_text}

# convert to dataframe
df_train = pd.DataFrame(train_data)
df_test = pd.DataFrame(test_data)

# save train and test data as csv files
df_train.to_csv('dataset_kyc_NI_train.csv', index=False)
df_test.to_csv('dataset_kyc_NI_test.csv', index=False)

# csv to text file conversion for training the file in fasttext
# train data
csv_file = 'Dataset/dataset_kyc_NI_train.csv'
txt_file = 'Dataset/dataset_kyc_NI_train.txt'
with open(txt_file, "w") as my_output_file:
    with open(csv_file, "r") as my_input_file:
        [my_output_file.write(" ".join(row) + '\n') for row in csv.reader(my_input_file)]
    my_output_file.close()

# test data
csv_file = 'Dataset/dataset_kyc_NI_test.csv'
txt_file = 'Dataset/dataset_kyc_NI_test.txt'
with open(txt_file, "w") as my_output_file:
    with open(csv_file, "r") as my_input_file:
        [my_output_file.write(" ".join(row) + '\n') for row in csv.reader(my_input_file)]
    my_output_file.close()
