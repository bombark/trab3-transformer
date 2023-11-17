from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier

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
    x_linha = []
    for i in range(49):
        x_linha.append(int(line[i]))
    x_data.append( x_linha )
    y_real.append( int(line[50]) )

print(y_real)

# Cria o SVM, treina e classifica
print("Treinando")
clf_svc = svm.SVC(
    C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, 
    probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, 
    max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None
    )

clf_nn = MLPClassifier (
    hidden_layer_sizes=(2,), activation='relu', solver='adam', alpha=0.0001, 
    batch_size='auto', learning_rate='constant', learning_rate_init=0.001, 
    power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, 
    verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, 
    early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, 
    epsilon=1e-08, n_iter_no_change=10, max_fun=15000
    )

clf = clf_svc

clf.fit(x_data, y_real)

print("Testando")
y_pred = clf.predict(x_data)

# Mostra o resultado
print(classification_report(y_real, y_pred))