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

fd = open("db_traducao.csv")
raw = fd.read().split("\n")
fd.close()
for _linha in raw:
    if len(_linha) == 0:
        continue
    linha = _linha.split(';')
    traducao[ linha[1] ] = linha[2]


# Cria um indice
index = {}

### le a base de dados em csv
with open('db_iso.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=';', quotechar='\"')
    corpus = list(reader)
    header, corpus = corpus[0], corpus[1:]

### Para cada linha coloca em seu respetivo lugar no indice e cria um dicionario de palavras
cont_palavras = 1
dicionario = {}
for linha in corpus:
    if len(linha) < 5:
        continue
    # print(linha)
    autor = linha[0].strip()
    titulo = linha[1].strip()
    assunto = linha[4].strip()

    if assunto in traducao:
        assunto = traducao[assunto]
    else:
        continue

    if assunto in index:
        index[assunto].append(linha)
    else:
        index[assunto] = [linha]

    autor_titulo = autor + ";" + titulo

    frase_tokens = word_tokenize(autor_titulo, language="portuguese")
    for token in frase_tokens:
        if token not in dicionario:
            dicionario[token] = cont_palavras
            cont_palavras += 1

# Gera o arquivo db_gen_indices com os indices usados e seu respectivo id
n_entrada = 50
fd_idx = open("gen_db_rotulos.csv", "w")
fd_data = open("gen_db_dados.csv", "w")
id_key = 1
for key,value in index.items():
    if len(value) < 10 :
        continue
    if len(key) < 5:
        continue

    fd_idx.write( f"{id_key};{key};{len(value)}\n" )
    # print("###", key) 
    for linha in value:
        autor = linha[0].strip()
        titulo = linha[1].strip()
        assunto = linha[4].strip()
        autor_titulo = autor + ";" + titulo

        frase_tokens = word_tokenize(autor_titulo, language="portuguese")
        if len(frase_tokens) > n_entrada:
            raise Exception("Aumentar o numero n_entrada")

        # gera a saida
        saida = ""
        for token in frase_tokens:
            saida += str(dicionario[token]) + ','

        for i in range(n_entrada-len(frase_tokens)):
            saida += '0,'
        saida += str(id_key)

        # salva a saida no arquivo
        fd_data.write(saida+"\n")
    id_key += 1

print("Gerado os arquivo gen_db_rotulos.csv e gen_db_dados.csv")
fd_idx.close()
fd_data.close()