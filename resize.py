import os
from PIL import Image

# save path
save_path = "/home/pengjie/SiT_0/sample_results/test_dresscode_512_384/"
input_path = "/home/pengjie/SiT_0/sample_results/test_dresscode/"
pathDir = os.listdir(input_path)

for i in pathDir:
    image = os.path.join(input_path, i)
    img = Image.open(image)
    width, height = img.size
    resized = img.resize((int(width * 0.5), int(height * 0.5)))
    resized.save(os.path.join(save_path, i))