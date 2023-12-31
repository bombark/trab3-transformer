from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

import numpy as np

tamanho_embedded = 3000

# Le os dados X e Y
print("Lendo os Dados")
x_data = []
fd = open(f"dados_pt_{tamanho_embedded}.csv")
raw = fd.read().split('\n')
fd.close()

x_data = np.zeros((len(raw), tamanho_embedded), dtype=np.float32)
y_real = np.zeros((len(raw)), dtype=np.uint32)

rows = 0
for _line in raw:
    line = _line.split(',')
    if len(line) < tamanho_embedded:
        continue
    x_linha = []
    for i in range(tamanho_embedded):
        x_data[rows][i] = float(line[i])
    y_real[rows] = int(line[tamanho_embedded])
    rows += 1

print(len(x_data))

# Divide em treino e teste
x_train, x_test, y_train, y_test = train_test_split(x_data, y_real, test_size=0.30)
print(len(x_train), len(x_test))

# Cria o SVM, treina e classifica
print("Treinando")
clf_svc = svm.SVC(
    C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, 
    probability=False, tol=0.001, cache_size=500, class_weight=None, verbose=False, 
    max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None
    )

clf_nn = MLPClassifier (
    hidden_layer_sizes=(50,500,500,50,), activation='tanh', solver='adam', alpha=0.0001,
    batch_size='auto', learning_rate='constant', learning_rate_init=0.00125,
    power_t=0.5, max_iter=10000, shuffle=True, random_state=None, tol=0.0001,
    verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
    epsilon=1e-05, n_iter_no_change=10, max_fun=15000
    )

clf = clf_svc


clf.fit(x_train, y_train)

print("Testando")
y_pred = clf.predict(x_test)

# print(y_test)
for i in range(len(y_pred)):
    print(y_pred[i], end=", ")
print()

# Mostra o resultado
print(classification_report(y_test, y_pred))

cont = 1
for i in range(len(y_pred)):
    if y_pred[i] == y_test[i]:
        cont += 1

print(f"Total: {len(y_pred)}, Acertos: {cont} ({cont/len(y_pred)}%)")

