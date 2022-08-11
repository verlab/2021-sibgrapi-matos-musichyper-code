from header import *

def prepare_oasis():

    folder = image_dataset_dir+"OASIS/"

    csv = "original/labels_orig.csv"

    df = pd.read_csv(folder+csv)

    names =  df["Theme"]
    valences = df["Valence_N"]
    arousals = df["Arousal_N"]

    all_list = []
    all_list.append(("filename","class"))

    for i in range(len(names)):
        filename = "images/"+names[i]+".jpg"

        valence = int(valences[i])
        arousal = int(arousals[i])

        valN_to_val = {
            100: 0,
            101: 0,
            102: 0,
            103: 1,
            104: 1,
            105: 2,
            106: 2,
            107: 3,
            108: 3,
        }

        arsN_to_ars = {
            100: 0,
            101: 0,
            102: 1,
            103: 2,
            104: 3,
        }

        valence = valN_to_val[valence]
        arousal = arsN_to_ars[arousal]

        quadrant = 0
        if(valence>1):
            if(arousal>1):
                quadrant=0
            else:
                quadrant=1
        else:
            if(arousal>1):
                quadrant=2
            else:
                quadrant=3
        
        all_list.append((filename,quadrant))

    fntmp1 = folder+"labels.csv"
    (pd.DataFrame(all_list)).to_csv(fntmp1, index=None, header=False)
    for item in all_list:
        print(item)

def prepare_gaped():
    folder = image_dataset_dir+"GAPED/"
    csv = "original/ALL.csv"
    df = pd.read_csv(folder+csv)
    file_class_list = []
    file_class_list.append(("filename","class"))
    for i in range(df.shape[0]):
        row = df.iloc[i,:].tolist()
        f,v,a = row[0],row[1],row[2]
        if(a>50):
            if(v>50):
                q=0
            elif(v<50):
                q=1
        else:
            if(v>50):
                q=3
            elif(v<50):
                q=2
        file_class_list.append((f,q))

    fntmp1 = folder+"labels.csv"
    (pd.DataFrame(file_class_list)).to_csv(fntmp1, index=None, header=False)
    for item in file_class_list:
        print(item)

def prepare_emomadrid():
    folder = image_dataset_dir+"EMMD/"
    csv = "original/labels_orig.csv"
    df = pd.read_csv(folder+csv)

    file_class_list = []
    file_class_list.append(("filename","class"))

    for i in range(df.shape[0]):
        row = df.iloc[i,:].tolist()
        f,v,a = row[0],row[1],row[2]
        f = "images/"+f+".jpg"
        if(a>1.2):
            if(v>1):
                q=0
            elif(v<1):
                q=1
        else:
            if(v>1):
                q=3
            elif(v<1):
                q=2
        file_class_list.append((f,q))

    fntmp1 = folder+"labels.csv"
    (pd.DataFrame(file_class_list)).to_csv(fntmp1, index=None, header=False)
    for item in file_class_list:
        print(item)

def prepare_iaps():
    folder = image_dataset_dir+"IAPS/"

    file_list = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".jpg"):
                file_list.append(os.path.join(root, file))

    file_class_list = []
    file_class_list.append(("video","class"))

    for f in file_list:
        i = -1
        while(f[i]!="/"):
            i-=1
        c = f[i-1]
        f = "images/"+str(c)+f[i:]
        file_class_list.append((f,c))

    fntmp1 = folder+"labels.csv"
    (pd.DataFrame(file_class_list)).to_csv(fntmp1, index=None, header=False)
    for item in file_class_list:
        print(item)

def prepare_pexels():
    folder = image_dataset_dir+"PEXELS/"

    file_list = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".jpg"):
                file_list.append(os.path.join(root, file))

    file_class_list = []
    file_class_list.append(("video","class"))
    for f in file_list:
        i = -1
        while(f[i]!="/"):
            i-=1
        c = int(f[i-1])
        f = "images/"+str(c)+f[i:]
        file_class_list.append((f,c))

    fntmp1 = folder+"labels.csv"
    (pd.DataFrame(file_class_list)).to_csv(fntmp1, index=None, header=False)
    for item in file_class_list:
        print(item)

def run():
    prepare_oasis()
    prepare_gaped()
    prepare_emomadrid()
    prepare_iaps()
    prepare_pexels()