def rgb_to_yuv(x):
    r, g, b = x[:, 0], x[:, 1], x[:, 2]

    Y = 0.299 * r + 0.587 * g + 0.114 * b
    U = -0.14713 * r - 0.28886 * g + 0.436 * b
    V = 0.615 * r - 0.51499 * g - 0.10001 * b

    return Y, U, V

def convert_img_range(img):
    # from [-1,1] to [0,1]
    cover = (img + 1) / 2
    return cover