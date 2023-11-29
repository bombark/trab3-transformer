import gensim.models
from nltk.tokenize import word_tokenize

model = "./ptwiki_20180420_100d.txt"
word_vectors = gensim.models.KeyedVectors.load_word2vec_format(model, binary=False)


fd = open("../dataset/mini_pt.csv")
raw = fd.read()
fd.close()

max_token = 30
code_size = 100
index = {}
cont_seq_assunto = 1

frase = 0
for line_ in raw.split("\n"):
    line = line_.split(';')
    if len(line) < 3:
        continue

    autor_titulo = line[0]+" "+line[1]
    tokens = word_tokenize(autor_titulo, language="portuguese")

    
    token_cont = 0
    for token in tokens:
        if token_cont >= max_token:
            break
        if token == ',' or token == "'":
            continue

        try:
            code = word_vectors[token.lower()]
            for i in range(code_size):
                print(code[i], end=",")
        except KeyError:
            for i in range(code_size):
                print("0.0", end=",")
        token_cont += 1

    for j in range(token_cont, max_token):
        for i in range(code_size):
            print("0.0", end=",")

    assunto = line[2]
    if assunto not in index:
        index[assunto] = cont_seq_assunto
        cont_seq_assunto += 1
    line[2] = index[assunto]

    print(line[2])

    frase += 1
    # if frase > 10:
    #    break
    