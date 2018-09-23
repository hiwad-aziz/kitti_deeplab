import numpy as np

CITYSCAPE_PALLETE = np.asarray([
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
    [0, 0, 0]], dtype=np.uint8)

width = 1242
height = 375

def logits2image(logits):
    logits = logits.astype(np.uint8)
    image = np.empty([height,width,3],dtype=float)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if(logits[i,j] == 255):
                image[i,j,:] = CITYSCAPE_PALLETE[19,:]
            else:
                image[i,j,:] = CITYSCAPE_PALLETE[logits[i,j],:]
    image = image.astype(np.uint8)
    return image
