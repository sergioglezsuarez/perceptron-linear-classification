import numpy as np


class Perceptron:
    """
    Clase que cuenta con dos métodos para entrenar un perceptrón y obtener predicciones.

    Atributos:
        pesos_ (array): vector de los pesos de las características del modelo entrenado (w).
        intercept_ (float): valor del término independiente (w0).
        max_iter (int): número máximo de iteraciones del proceso de entrenamiento.
        mezclar (bool): booleano que indica si las muestras se mezclan entre cada iteración o no.
        eta0 (float): la fracción de la muestra mal clasificada por la que se actualiza el vector de pesos.
        semilla (int): número que indica el valor de la semilla en caso de mezclar aleatoriamente las muestras.
    """
    def __init__(self, max_iter=1000, mezclar=True, eta0=1.0, semilla=0):
        """
        Función constructora de la clase.
        """

        assert type(max_iter) == int and max_iter > 0, "max_iter debe ser un entero mayor que 0."
        assert type(mezclar) == bool, "mezclar debe ser de tipo booleano."
        assert type(eta0) == int or type(eta0) == float, "eta0 debe ser un número real."
        assert semilla is None or type(semilla) == int, "Si mezclar es True, semilla debe ser un entero."
        self.pesos_ = None
        self.intercept_ = None
        self.max_iter = max_iter
        self.mezclar = mezclar
        self.eta0 = eta0
        self.semilla = semilla

    def fit(self, X, y):
        """
        Función que entrena un modelo de perceptrón.

        :param X: (ndarray) numpy array bidimensional que contiene las características de las muestras a clasificar.
        :param y: (array-like) iterable unidimensional que contiene la clase a la que pertenece cada muestra X.
        :return: (Perceptron) devuelve una instancia de la clase Perceptron con los pesos correspondientes al modelo entrenado.
        """

        assert type(X).__module__ == "numpy", "El argumento X debe ser un numpy array."
        assert len(X) > 0, "El argumento X debe tener como mínimo 1 muestra."
        assert all([len(x) > 0 for x in X]), "Las muestras deben tener una característica como mínimo."
        assert len(X) == len(y), "X e y deben tener la misma cantidad de muestras."

        if self.semilla is not None:
            np.random.seed(self.semilla)
        self.pesos_ = np.array([0] * len(X[0]))
        self.intercept_ = 0

        for _ in range(self.max_iter):
            xcount = 0  # contador del número de muestras que son correctamente clasificadas
            for i in range(len(X)):
                f = self.intercept_ + np.dot(self.pesos_, X[i])
                if y[i] == 1 and f < 0:
                    self.intercept_ = self.intercept_ + self.eta0
                    self.pesos_ = [self.pesos_[i2] + self.eta0 * X[i][i2] for i2 in range(len(self.pesos_))]
                elif y[i] == 0 and f >= 0:
                    self.intercept_ = self.intercept_ - self.eta0
                    self.pesos_ = [self.pesos_[i2] - self.eta0 * X[i][i2] for i2 in range(len(self.pesos_))]
                else:
                    xcount += 1
            if xcount == len(X):
                break
            if self.mezclar:
                p = np.random.permutation(len(X))
                X = np.array(X)[p].tolist()
                y = np.array(y)[p].tolist()

        return self

    def predict(self, X):
        """
        Función que predice a qué clase pertenecen las muestras X pasadas.

        :param X: (ndarray) numpy array bidimensional que contiene las características de las muestras a clasificar.
        :return: (list) lista unidimensional que contiene las clases de las predicciones.
        """

        assert type(X).__module__ == "numpy", "El argumento X debe ser un numpy array."
        assert len(X) > 0, "El argumento X debe tener como mínimo 1 muestra."
        assert all([len(x) > 0 for x in X]), "Las muestras deben tener una característica como mínimo."

        preds = []
        for i in range(len(X)):
            f = self.intercept_ + np.dot(self.pesos_, X[i])
            preds.append(f)
        preds = [0 if pred < 0 else 1 for pred in preds]
        return preds