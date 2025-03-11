import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.utils import resample

dataset = pd.read_csv("practica1\metodosDeValidacion.csv")

print(dataset)

#Retorna el conjunto de entrenamiento, de prueba y las etiquetas de lo elementos de estos conjuntos
entrenamiento, prueba, etiqueta_e, etiqueta_p = train_test_split(dataset.x, dataset.y, test_size=0.3, train_size=0.7, random_state=None, shuffle=False, stratify=None)

print("\nConjunto de entrenamiento\n", entrenamiento)
print("\nEtiquetas de conjunto de entrenamiento\n", etiqueta_e)
print("\nConjunto de prueba\n", prueba)
print("\nEtiquetas de conjunto de prueba\n", etiqueta_p)

#kfold no divide directamente el conjunto de entrenamiento, si no que genera indices que se tienen que recorrer manualmente para hacer los pliegues
kf = KFold(n_splits=6, shuffle=False)

for indice_entreno, indice_validacion in kf.split(entrenamiento):
    print("\nIndices para entreno\n", indice_entreno)
    print("\nIndices para validacion\n", indice_validacion)
    x_entreno, x_validacion = entrenamiento[indice_entreno], entrenamiento[indice_validacion]
    y_entreno, y_validacion = etiqueta_e[indice_entreno], etiqueta_e[indice_validacion]
    print("\nConjunto de entrenamiento\n", x_entreno)
    print("\nConjunto de validacion\n", x_validacion, "\n\n")

#LeaveOneOut funciona igual que k fold, ya que es un kfold con n pliegues, por esto, no recibe parametros, ya que esta predefinido su comportamiento
loo = LeaveOneOut()

for indice_entreno, indice_validacion in loo.split(entrenamiento):
    print("\nIndices para entreno\n", indice_entreno)
    print("\nIndices para validacion\n", indice_validacion)
    x_entreno, x_validacion = entrenamiento[indice_entreno], entrenamiento[indice_validacion]
    y_entreno, y_validacion = etiqueta_e[indice_entreno], etiqueta_e[indice_validacion]
    print("\nConjunto de entrenamiento\n", x_entreno)
    print("\nConjunto de validacion\n", x_validacion, "\n\n")

def bootstrap_sampling(entrenamiento, etiqueta_e, n_samples, random_state = 42, replace = True):
    entrenamiento = np.array(entrenamiento).reshape(-1, 1)
    etiqueta_e = np.array(etiqueta_e)
    x_boot, y_boot = resample(entrenamiento, etiqueta_e, replace=replace, n_samples=n_samples, random_state=random_state)
    # Encontrar las muestras que no están en el conjunto de entrenamiento bootstrap
    validacion = np.array([sample.tolist() not in x_boot.tolist() for sample in entrenamiento])
    X_val = entrenamiento[validacion]
    print("Conjunto de entrenamiento:")
    print(x_boot)
    print("Conjunto de validación:")
    print(X_val)

bootstrap_sampling(entrenamiento, etiqueta_e, 9, 42, True)
bootstrap_sampling(entrenamiento, etiqueta_e, 9, 43, True)
