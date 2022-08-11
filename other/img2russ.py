from header import *

def get_image_names():
    entries = os.listdir(folder)
    image_names = []
    for name in entries:
        if ".jpg" in name:
            image_names.append(name)
    return(image_names)

def get_brightness(img):
    sum = 0
    rows = len(img)
    columns = len(img[0])
    for i in range(rows):
        for j in range(columns):
            rgb_max = max(img[i][j][0],img[i][j][1],img[i][j][2])
            sum = sum + rgb_max
    return sum/(rows*columns)

def get_saturation(img):
    sum = 0
    rows = len(img)
    columns = len(img[0])
    for i in range(rows):
        for j in range(columns):

            r = img[i][j][0]
            g = img[i][j][1]
            b = img[i][j][2]
            
            max_rgb = max(r,g,b)
            dist = math.sqrt(math.pow((max_rgb-r),2)+math.pow((max_rgb-g),2)+math.pow((max_rgb-b),2))
            sum = sum + dist

    return sum/(rows*columns)

def get_arousal_valence(image_path):
    img = cv2.imread(image_path) 
    average_brightness = get_brightness(img)
    average_saturation = get_saturation(img)
    arousal = -31*average_brightness + 60*average_saturation
    valence = 69*average_brightness + 22*average_saturation  
    return arousal, valence

def run(image_filename):
    print(get_arousal_valence(image_filename))