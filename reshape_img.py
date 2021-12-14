def img_reshape(folder_in, folder_out, folder_main, format, size):

    '''
    Function that resizes the image creating a copy for easier access.

    Parameters:

        Folder layout:

            -->folder_proyect
                             --> folder_main
                                            --> folder_in
                                            --> folder_out

        format: select img format (example: jpg, png) without the "."

        size: size to which we want to reduce
            
    '''



    import time
    from glob import glob
    import os
    import cv2



    st = time.time()

    cwsinput = folder_in +'\*.{format}'
    img_input_dir = cwsinput
    files = glob(img_input_dir)
    print(len(files))

    cwdoutput = os.getcwd() + folder_out
    img_output_dir =cwdoutput
    imagesize = (size,size)
    for i,f1 in enumerate(files):
        startindex = f1.find(folder_main)
        tempf1 = f1[startindex:]
        tempfn = img_output_dir + tempf1
        inputimage = cv2.imread(f1)
        sizedimage = cv2.resize(inputimage, imagesize)
        n = str(i) 
        cv2.imwrite(os.path.join(folder_out, ('image'+ n +'.{format}')), sizedimage)
        
    print('The program took %s seconds' %(time.time()-st))