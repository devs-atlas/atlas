# get mapping from integers to chars -> decode
itoc = {i: c for i, c in enumerate(vocab)}
# get mapping from chars to integers -> encode
ctoi = {c: i for i, c in enumerate(vocab)}

# function to map char to integer for all chars in input string x
encode = lambda x: [ctoi[c] for c in x]

# function to map integer to char for all chars in input string x
decode = lambda x: [itoc[i] for i in x]

encode("hello"), decode(encode("hello"))

# SNIPPET #
([46, 43, 50, 50, 53], ['h', 'e', 'l', 'l', 'o'])
