import numpy as np
import os
import xml.etree.ElementTree as ET
import cv2 
import time
from dask import delayed, compute

class Parser:
    def __init__(self, input_path:str):
        if not os.path.isdir(input_path):
            # if folder doesn't exits, then throw an error
            pass
        else:
            # if floder is empty, then throw error
            pass

        # set folder_path
        self.input_path = input_path

    def parse(self, output_path:str = None):

        start_time = time.time()

        if not output_path:
            # if output_path is not specified, then take default one
            self.output_path = os.path.join(os.getcwd(),'parsed_data')
            if not os.path.isdir(self.output_path):
                # if folder doesnt exits, then create it
                os.mkdir(self.output_path)
        
        # get different class names from xml files
        xml_files = []
        for x in [x for x in os.listdir(self.input_path)]:
            if x.endswith('.xml'):
                xml_files.append(x)
        xml_files = np.array(xml_files)

        # create list of class names
        cl = []

        for x in xml_files:
            tree = ET.parse(os.path.join(self.input_path, x))
            root = tree.getroot()
            for obj in root.findall('object'):
                cl.append(obj.find('name').text)
        cl = sorted(set(cl))
        # cl.remove('Schraube_gespiegelt')

        class_list = np.array(cl)


        # create new folders in the output folder
        for x in class_list:
            if not os.path.isdir(os.path.join(self.output_path, x)):
                os.mkdir(os.path.join(self.output_path, x))
            
        @delayed
        def extract(self,x):
            # Extract required data from xml file
            tree = ET.parse(os.path.join(self.input_path, x))
            root = tree.getroot()
            fn = root.find('filename').text
            img = cv2.imread(os.path.join(self.input_path,fn))

            objects = root.findall('object')
            # sort them alphabetically
            objects = sorted(objects, key=lambda x: x.find('name').text)
            count = 0
            n = 0
            
            for obj in objects:
                
                # find number of objects with same name tag
                if count == 0:
                    #  get class name and bounding box
                    class_name = obj.find('name').text
                    n = len([x for x in objects if x.find('name').text == class_name])

                bbx = obj.find('bndbox')
                xmin = int(bbx.find('xmin').text)
                ymin = int(bbx.find('ymin').text)
                xmax = int(bbx.find('xmax').text)
                ymax = int(bbx.find('ymax').text)

                # crop the image
                cropped_img = img[ymin:ymax, xmin:xmax]
                # cv2.imshow(class_name, cropped_img)
                # cv2.waitKey(0)

                # save the cropped image in respective folder
                if n == 1:
                    cv2.imwrite(
                        os.path.join(self.output_path, class_name, f'{class_name}_{fn}'),
                        cropped_img)
                else:
                    count += 1
                    cv2.imwrite(
                        os.path.join(self.output_path, class_name, f'{class_name}_{count}_{fn}'),
                        cropped_img)
                    
                    if n == count :
                        count = 0
    
        y = []
        for x in xml_files:
            y.append(delayed(extract(self, x)))

        compute(y)
                
            

        end_time = time.time()
        print (f'time taken to parse: {end_time - start_time}s')


if __name__ == '__main__':
    p = Parser (os.path.join(os.getcwd(),'data'))
    p.parse()
