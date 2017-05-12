from PIL import Image
import os
import numpy as np
folder = os.path.abspath(os.path.join("./", os.pardir)+"/MTFL") + "/"
finalSize = 150

counter = 0
infoFiles = ["training.txt", "testing.txt"]
for idx in range(len(infoFiles)):
    info = infoFiles[idx]
    f = open(folder+info,"r")
    fnew = open("tmp", 'a')
    if idx == 0:
        fnew_augmented = open("aug_"+info,'a')
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
            # Skip non-square images
            img.close()
            continue

        # Resize and check RGB
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

        coords = []
        coordsScaleFactor = float(finalSize) / float(originalWidth)
        for i in range(1,11):
            coords.append(float(line[i])*coordsScaleFactor)

        
        attributes = np.array([int(line[i]) for i in range(11,15)])

        # Write resized img to file
        fnew.write(imgName)
        for coord in coords:
            fnew.write(" " + str(coord))
        for attribute in attributes:
            fnew.write(" " + str(attribute - 1)) # Subtract 1 for better indexing
        fnew.write("\n")

        # Mirror the image if it's not part of test data
        if idx == 0:
            # Get the new img name
            splitName = imgName.split('.')
            imgNameTransp = splitName[0] + '_transl.' + splitName[1]

            # Mirror the image and save it
            imgTransp = img.copy().transpose(Image.FLIP_LEFT_RIGHT)
            imgTransp.save(folder+imgNameTransp)
            imgTransp.close()

            coordsTransp = [0 for i in range(10)]
            # Translate x-coords for eyes, nose, and mouth
            coordsTransp[0] = 150 - coords[1]
            coordsTransp[1] = 150 - coords[0]
            coordsTransp[2] = 150 - coords[2]
            coordsTransp[3] = 150 - coords[4]
            coordsTransp[4] = 150 - coords[3]
            # Translate y-coords for eyes, nose, and mouth
            coordsTransp[5] = coords[6]
            coordsTransp[6] = coords[5]
            coordsTransp[7] = coords[7]
            coordsTransp[8] = coords[9]
            coordsTransp[9] = coords[8]
            # Translate attributes
            attributesTransp = np.array([int(line[i]) for i in range(11,15)])
            attributesTransp[3] = 6 - attributesTransp[3] # Translate head pose

            # Write resized old img to augmented file
            fnew_augmented.write(imgName)
            for coord in coords:
                fnew_augmented.write(" " + str(coord))
            for attribute in attributes:
                fnew_augmented.write(" " + str(attribute - 1))
            fnew_augmented.write("\n")

            # Write mirrored img to augmented file
            fnew_augmented.write(imgNameTransp)
            for coord in coordsTransp:
                fnew_augmented.write(" " + str(coord))
            for attribute in attributesTransp:
                fnew_augmented.write(" " + str(attribute - 1))
            fnew_augmented.write("\n")

        # Save resized img
        img.save(folder+imgName)
        img.close()
        
        counter = counter + 1
        if counter % 1000 == 0:
            print(counter,"files processed")
    
    f.close()
    fnew.close()
    if idx == 0:
        fnew_augmented.close()
    os.remove(folder+info)
    os.rename("tmp",folder+info)
    