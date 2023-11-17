from sklearn import svm
from sklearn.metrics import classification_report

# Le os dados X e Y
print("Lendo os Dados")
x_data = []
y_real = []
fd = open("gen_db_dados.csv")
raw = fd.read()
fd.close()
for _line in raw.split('\n'):
    line = _line.split(',')
    if len(line) < 50:
        continue
    x_data.append( line[0:49] )
    y_real.append( line[50] )

# Cria o SVM, treina e classifica
print("Treinando")
clf = svm.SVC()
clf.fit(x_data, y_real)

print("Testando")
y_pred = clf.predict(x_data)

# Mostra o resultado
print(classification_report(y_real, y_pred))