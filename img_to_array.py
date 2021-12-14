def img_to_array(folder_in, format, max_elem=7000):

    '''
    Function that converts the image into an array.

    Parameters:

        folder_in: Folder where the images are

        format: Select img format (example: jpg, png) without the "."

        max_elem: Maximum number of images to convert, default 7000
            
    '''

    import time
    from glob import glob
    import cv2
    import numpy as np


    st = time.time()
    cwdinput = folder_in + '\*.' + format
    img_input_dir = cwdinput
    files = glob(img_input_dir)
    print(len(files))


    X_data = []
    iterations = 0
    for myFile in files:
        image = cv2.imread(myFile)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        X_data.append(image)
        iterations = iterations + 1
        if(iterations % 1000 == 0):
            print(iterations)
        if(iterations == max_elem):
            break

    X_data_array = np.asarray(X_data)
    print('X_data shape: ', X_data_array.shape)

    print('The program took %s seconds ' %(time.time()-st))

    return X_data_array