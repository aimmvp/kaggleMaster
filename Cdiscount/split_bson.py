# https://www.kaggle.com/hvrigazov/splitting-the-bson-file

from tqdm import tqdm_notebook
import bson
import numpy as np
import io
import matplotlib.pyplot as plt
from PIL import Image

# num_images = 12371293 # total : 7,069,896
num_images = 7069896
num_points = 100000

checkpoints = np.linspace(0, num_images, num_points, dtype=np.int64)
file_pointers = [0]

bar = tqdm_notebook(total=num_images)
i = 0
current_checkpoint = 0
with open('./input/train.bson', 'rb') as fbson:
    data = bson.decode_file_iter(fbson)

    for c, d in enumerate(data):
        category = d['category_id']
        _id = d['_id']
        for e, pic in enumerate(d['imgs']):
            i += 1
            bar.update()

        if i > checkpoints[current_checkpoint + 1] and i < checkpoints[current_checkpoint + 2] :
            file_pointers.append(fbson.tell())
            current_checkpoint += 1

print("#####################################")
print("length [", len(file_pointers), "]")
print(file_pointers[0])
print(file_pointers[1])
