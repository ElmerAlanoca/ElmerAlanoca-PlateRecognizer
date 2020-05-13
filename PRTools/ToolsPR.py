import cv2
import pytesseract
import matplotlib.pyplot as plt


class ImageStorage:
    '''Nos permite cargar una imagen'''
    
    @staticmethod
    def read_image(path_img):
        '''Leer una imagen desde el disco y devolver in objeto imagen'''
        if isinstance(path_img, str):
            img = cv2.imread(path_img)
            return img
        else:
            print("formato no valido")
            return None


class PlateRecognizer:
    '''Nos permite obtener la placa de una fotografia de un automovil en forma de String'''

    @staticmethod
    def plate_recognizer(image):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
        blur = cv2.bilateralFilter(gray, 11,90, 90)
    
        edges = cv2.Canny(blur, 30, 200)
    
        cnts, new = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
        image_copy = image.copy()
    
        _ = cv2.drawContours(image_copy, cnts, -1, (255,0,255),2)
    
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
    
        image_copy = image.copy()
    
        _ = cv2.drawContours(image_copy, cnts, -1, (255,0,255),2)

        plate = None
        
        for c in cnts:
            perimeter = cv2.arcLength(c, True)
            edges_count = cv2.approxPolyDP(c, 0.02 * perimeter, True)
            if len(edges_count) == 4:
                x,y,w,h = cv2.boundingRect(c)
                plate = image[y:y+h, x:x+w]
                break

        text = pytesseract.image_to_string(plate, lang="eng")

        return text


class SavePlate:
    '''Permite guardar el valor de la placa en un archivo csv'''

    @staticmethod
    def save_plate_as_txt(text, path2save):
        name_file = path2save + '/placas.txt'

        with open(name_file, 'a') as f:
            f.write('Placa: ' + text + '\n')