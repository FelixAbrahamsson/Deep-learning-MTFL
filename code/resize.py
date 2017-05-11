from PIL import Image
import os
import numpy as np
folder = os.path.abspath(os.path.join("./", os.pardir)+"/MTFL") + "/"
finalSize = 150

counter = 0
for info in ["training.txt", "testing.txt"]:
    f = open(folder+info,"r")
    fnew=open("tmp", 'a')
    lines = f.readlines()
    for line in lines:
        
        line = line.strip("\n ").split(" ")
        imgName = line[0].replace("\\", "/")
        
        if imgName == "":
            break
        
        img = Image.open(folder+imgName, mode='r')
        originalWidth = img.width
        originalHeight = img.height
        if (originalWidth != originalHeight):
            img.close()
            continue

        if(originalWidth != finalSize):
            img = img.resize((finalSize, finalSize), Image.ANTIALIAS)

            # Check if img is RGB or greyscale
            pixels = list(img.getdata())
            width, height = img.size
            pixelsArray = np.asarray([pixels[i * width:(i + 1) * width] for i in range(height)])
            if len(pixelsArray.shape) != 3:
                # img is greyscale, skip it
                img.close()
                continue

            img.save(folder+imgName)
            img.close()

        coords = []
        coordsScaleFactor = float(finalSize) / float(originalWidth)
        
        for i in range(1,11):
            coords.append(float(line[i])*coordsScaleFactor)
            
        attributes = np.array([int(line[i]) for i in range(11,15)])
        
        fnew.write(imgName)
        for coord in coords:
            fnew.write(" " + str(coord))
        for attribute in attributes:
            fnew.write(" " + str(attribute - 1))
        fnew.write("\n")
        
        counter = counter +1
        if counter % 1000 == 0:
            print(counter,"files resized")
    
    f.close()
    fnew.close()
    os.remove(folder+info)
    os.rename("tmp",folder+info)
    