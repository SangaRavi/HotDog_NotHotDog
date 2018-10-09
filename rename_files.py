
directory = '' #Set directory to rename the files for training

import os

directoryName = directory
filePath = os.path.abspath(directoryName)
filePathWithSlash = filePath + "\\"

for counter, filename in enumerate(os.listdir(directoryName)):

    filenameWithPath = os.path.join(filePathWithSlash, filename)

    #change as respective with the required file name and format
    os.rename(filenameWithPath, filenameWithPath.replace(filename,"Not_HotDog." + \
          str(counter).zfill(4) + ".jpg" ))