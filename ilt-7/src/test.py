import sys
print(f"Python version: {sys.version}")

try:
    import numpy as np
    print("numpy version:", np.__version__)
except:
    print("numpy is not installed")

try:
    import pandas as pd
    print("pandas version:", pd.__version__)
except:
    print("pandas is not installed")

try:
    import PIL
    print("Pillow version:", PIL.__version__)
except:
    print("Pillow is not installed")

try:
    import scipy
    print("scipy version:", scipy.__version__)
except:
    print("scipy is not installed")

try:
    import tensorflow as tf
    print("tensorflow version:", tf.__version__)
except:
    print("tensorflow is not installed")

try:
    import tensorflow_datasets as tfds
    print("tfds version:", tfds.__version__)
except:
    print("tensorflow-datasets is not installed")

try:
    lib_version = [
        np.__version__=="1.24.3",
        pd.__version__=="2.0.3",
        PIL.__version__=="10.0.0",
        scipy.__version__=="1.10.1",
        tf.__version__=="2.13.0",
        tfds.__version__=="4.9.2"
    ]

    if all(lib_version):
        print("All main dependencies are ready!")
except:
    pass