# =======================================================================================
#  Header
# =======================================================================================

import csv
from nltk.tokenize import word_tokenize

# =======================================================================================
#  Main
# =======================================================================================

# cria a tabela de traducao a partir do arquivo
traducao = {}

"""
fd = open("db_traducao.csv")
raw = fd.read().split("\n")
fd.close()
for _linha in raw:
    if len(_linha) == 0:
        continue
    linha = _linha.split(';')
    traducao[ linha[1] ] = linha[2]
"""

# Cria um indice
index = {}

### le a base de dados em csv
with open('../dataset/mini_database.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=';', quotechar='\"')
    corpus = list(reader)
    header, corpus = corpus[0], corpus[1:]

### Para cada linha coloca em seu respetivo lugar no indice e cria um dicionario de palavras
cont_palavras = 1
cont_assunto = 0
cont_seq_assunto = 1
dicionario = {}
for linha in corpus:
    if len(linha) < 3:
        continue
    # print(linha)
    autor = linha[0].strip()
    titulo = linha[1].strip()
    assunto = linha[2].strip()

    if assunto not in index:
        index[assunto] = cont_seq_assunto
        cont_seq_assunto += 1
    linha[2] = index[assunto]

    autor_titulo = autor + ";" + titulo
    frase_tokens = word_tokenize(autor_titulo, language="portuguese")
    for token in frase_tokens:
        if token not in dicionario:
            dicionario[token] = cont_palavras
            cont_palavras += 1


    

# Gera o arquivo db_gen_indices com os indices usados e seu respectivo id
n_entrada = 60

fd_data = open("gen_db_dados.csv", "w")
id_key = 1
for linha in corpus:
    if len(linha) < 3:
        continue

    # fd_idx.write( f"{id_key};{key};{len(value)}\n" )
    # print("###", key) 
    autor = linha[0].strip()
    titulo = linha[1].strip()
    assunto = linha[2]
    autor_titulo = autor + ";" + titulo

    frase_tokens = word_tokenize(autor_titulo, language="portuguese")
    if len(frase_tokens) > n_entrada:
        continue
        # raise Exception("Aumentar o numero n_entrada")

    # gera a saida
    saida = ""
    for token in frase_tokens:
        saida += str(dicionario[token]) + ','

    for i in range(n_entrada-len(frase_tokens)):
        saida += '0,'
    saida += str(assunto)

    # salva a saida no arquivo
    fd_data.write(saida+"\n")

    # print(f"{autor_titulo};{assunto}")

print("Gerado os arquivo gen_db_rotulos.csv e gen_db_dados.csv")

fd_data.close()


fd_idx = open("gen_db_rotulos.csv", "w")
for key,value in index.items():
    fd_idx.write(f"{value};{key}\n")
fd_idx.close()