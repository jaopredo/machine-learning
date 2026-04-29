import numpy as np

def resizing(image, new_width, new_height):
    old_width, old_height = image.shape[1], image.shape[0]

    new_image = np.zeros((new_height, new_width), dtype=image.dtype)

    for i in range(new_height):
        for j in range(new_width):
            # Calculando onde os pixels da imagem atual cairiam na imagem original
            s = (i + .5)*old_height/new_height - .5
            t = (j + .5)*old_width/new_width - .5

            s0 = int(np.floor(s))
            t0 = int(np.floor(t))
            s1 = s0 + 1
            t1 = t0 + 1

            neighbours = [
                (s0, t0),
                (s0, t1),
                (s1, t0),
                (s1, t1)
            ]

            ds = [s - s0, s - s1]
            dt = [t - t0, t - t1]

            pixel_value = 0
            for k in range(4):
                s_k, t_k = neighbours[k]
                if s_k < 0 or s_k >= old_height or t_k < 0 or t_k >= old_width:
                    continue

                pixel_value += image[s_k, t_k] * (1 - abs(ds[k//2])) * (1 - abs(dt[k%2]))
            
            new_image[i, j] = pixel_value
    return new_image