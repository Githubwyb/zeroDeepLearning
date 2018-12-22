import zipfile
import os

for dirpath,dirnames,filenames in os.walk('./'):
    if dirpath == './':
        for folderName in dirnames:
            print('Begin zip', folderName)
            # z = zipfile.ZipFile(folderName + '.zip', 'w', zipfile.ZIP_STORED)
            for dp, dn, fn in os.walk(folderName):
                for fileName in fn:
                    print(folderName + '/' + fileName)
            # z.close()