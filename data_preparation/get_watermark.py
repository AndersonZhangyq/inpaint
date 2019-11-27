from PIL import Image

origin = Image.open('/home/zyq/code/inpaint/data_preparation/download.jpg')
mask = Image.open('/home/zyq/code/inpaint/data_preparation/1_dip_2.png').resize(origin.size)

for x in range(mask.size[0]):
    for y in range(mask.size[1]):
        pixel = mask.getpixel((x, y))
        if pixel != (255,255,255,128) and pixel[-1] != 0:
            mask.putpixel((x, y), (255,255,255,128))


output = Image.alpha_composite(origin.convert("RGBA"), mask)

mask = mask.convert('RGB')
for x in range(mask.size[0]):
    for y in range(mask.size[1]):
        pixel = mask.getpixel((x, y))
        if pixel == (255,255,255):
            mask.putpixel((x, y), (0,0,0))
        elif pixel == (0,0,0):
            mask.putpixel((x, y), (255,255,255))
mask.save("mask.png")
output.save('saved_mask.png')