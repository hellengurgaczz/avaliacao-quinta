from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

app = Flask(__name__)

# Carregar o conjunto de dados do Titanic
url = 'https://learnenough.s3.amazonaws.com/titanic.csv'
titanic = pd.read_csv(url)

# Limpar o conjunto de dados
columns_to_drop = ['Name', 'PassengerId', 'Cabin', 'Embarked', 'SibSp', 'Parch', 'Ticket', 'Fare']

for column in columns_to_drop:
    titanic = titanic.drop(column, axis=1)

for column in ['Age', 'Sex', 'Pclass']:
    titanic = titanic[titanic[column].notna()]

sex_int = {'male': 0, 'female': 1}
titanic['Sex'] = titanic['Sex'].map(sex_int)

# Dividir os dados em recursos (X) e rótulo (y)
X = titanic.drop('Survived', axis=1)
y = titanic['Survived']

@app.route('/')
def index():
    classifiers = ['KNeighborsClassifier', 'SVC', 'MLPClassifier', 'DecisionTreeClassifier', 'RandomForestClassifier']
    return render_template('index.html', classifiers=classifiers)

@app.route('/train', methods=['POST'])
def train():
    selected_classifier = request.form['classifier']

    # Dividir o conjunto de dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    # Configurar o classificador selecionado
    if selected_classifier == 'KNeighborsClassifier':
        neighbors = int(request.form['neighbors'])
        clf = KNeighborsClassifier(n_neighbors=neighbors)
    elif selected_classifier == 'SVC':
        clf = SVC()
    elif selected_classifier == 'MLPClassifier':
        max_iter = int(request.form['max_iter'])
        clf = MLPClassifier(max_iter=max_iter)
    elif selected_classifier == 'DecisionTreeClassifier':
        max_depth = int(request.form['max_depth'])
        clf = DecisionTreeClassifier(max_depth=max_depth)
    elif selected_classifier == 'RandomForestClassifier':
        n_estimators = int(request.form['n_estimators'])
        clf = RandomForestClassifier(n_estimators=n_estimators)
    else:
        return "Classificador não reconhecido"

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1_score = f1_score(y_test, y_pred, average='macro')

    # Calcular a matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    cm_img_buf = io.BytesIO()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Previsto')
    plt.ylabel('Atual')
    plt.title('Matriz de Confusão')
    plt.tight_layout()
    plt.savefig(cm_img_buf, format='png')
    cm_img_buf.seek(0)
    cm_img_base64 = base64.b64encode(cm_img_buf.read()).decode('utf-8')
    plt.close()

    return render_template('index.html', classifier=selected_classifier, accuracy=accuracy,
                           macro_f1_score=macro_f1_score, confusion_matrix=cm_img_base64)

if __name__ == '__main__':
    app.run(debug=True)
