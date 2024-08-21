from PIL import Image
from PIL import features

# print(features.pilinfo())
print (features.check_module('webp'))

im = Image.open(r"C:\Users\Chris\Downloads\images\a.webp")
im.save("2.png", "png")

# from webptools import dwebp
# print(dwebp(r"C:\Users\Chris\Downloads\images\2.webp",r"C:\Users\Chris\Downloads\images\2.png","-o", "-v"))

# import webp
# # Load an image
# img = webp.load_image(r"C:\Users\Chris\Downloads\images\2.webp")
# webp.save_image(img, 'image.png')