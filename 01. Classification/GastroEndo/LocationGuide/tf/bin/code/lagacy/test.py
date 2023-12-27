from sklearn.utils import resample

import random
import numpy as np

if __name__ == '__main__':
    a = [1,2,3,4,5,6,7,8,9,10]
    b = ['a','b','c','d','e','f','g','h','i','j']
    random_seed = random.randint(0,10000)

    rs0 = resample(a, n_samples=1, replace=True, stratify=None,
                random_state=random_seed)
    rs1 = resample(b, n_samples=1, replace=True, stratify=None,
                random_state=random_seed)

    print(rs0)
    print(rs1)