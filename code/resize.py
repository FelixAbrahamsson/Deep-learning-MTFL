from PIL import Image
import os
import numpy as np
folder = os.path.abspath(os.path.join("./", os.pardir)+"/MTFL") + "/"
counter = 0
for info in ["training.txt", "testing.txt"]:
    f = open(folder+info,"r")
    fnew=open("tmp", 'a')
    lines = f.readlines()
    for line in lines:
        
        line = line.strip("\n ").split(" ")
        imgName = line[0].replace("\\", "/")
        
        if imgName[0:3] == "net":
            continue
        if imgName == "":
            break
        
        img = Image.open(folder+imgName, mode='r')
        originalWidth = img.width
        if(originalWidth != 150):
            img = img.resize((150, 150), Image.ANTIALIAS)
            pixels = list(img.getdata())
            width, height = img.size
            img.save(folder+imgName)
            img.close()

        coords = []
        coordsScaleFactor = float(150) / float(originalWidth)
        
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
    
    fnew.close()
    os.rename("tmp",folder+info)
    
