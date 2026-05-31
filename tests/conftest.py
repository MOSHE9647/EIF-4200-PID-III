import os
import sys

# Añade la raíz del repositorio al inicio de sys.path para que 'src' sea importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
