tokenized_data = encode(data)  # run encoding function on entire dataset
n = int(len(tokenized_data) * 0.9)  # define training slice as 90%
train_data = tokenized_data[:n]  # get first 90% of data
test_data = tokenized_data[n:]  # get last 90% of data
