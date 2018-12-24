import zipfile
import os

z = zipfile.ZipFile('dirName.zip', 'w', zipfile.ZIP_STORED)
for dirPath, dirNames, fileNames in os.walk('1.Perceptron'):
    for fileName in fileNames:
        z.write(dirPath + '/' + fileName)
        print(dirPath + '/' + fileName)

    for dirName in dirNames:
        z.write(dirPath + '/' + dirName)
        print(dirPath + '/' + dirName + '/')
z.close()
