import os
import sys

def verificar_ambiente():
    # Verificar si está en Google Colab
    if 'google.colab' in sys.modules:
        return "Estás trabajando en Google Colab."
    # Verificar si está en un sistema Windows
    elif os.name == 'nt':
        print("Estás trabajando en un sistema Windows.")
        return 'win'
    else:
        print("El ambiente de trabajo no es Google Colab ni Windows.")
        return 'colab'