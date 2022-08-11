from header import *

lang_groups = [
        "arabic",
        "chinese",
        "dutch",
        "english",
        "french",
        "german",
        "italian",
        "persian",
        "polish",
        "russian",
        "spanish",
        "turkish",   
    ]

anp_util_lim = 0.4

def reindex_anp_lists_for_lang_group(lang_group):

    print("Lang group: "+str(lang_group))

    folder = image_dataset_dir+"MVSO/"

    urls_csv = folder+"original/anp_url/"+lang_group+".csv"
    emots_csv = folder+"original/anp_emot/ANP_emotion_mapping_"+lang_group+".csv"

    nanp_url_dir = folder+"original/nanp_url/"
    nanp_emot_dir = folder+"original/nanp_emot/"

    if not(os.path.exists(nanp_url_dir)):
        os.makedirs(nanp_url_dir)
    if not(os.path.exists(nanp_emot_dir)):
        os.makedirs(nanp_emot_dir)

    urls_csv_out = nanp_url_dir+lang_group+".csv"
    emots_csv_out = nanp_emot_dir+"ANP_emotion_mapping_"+lang_group+".csv"

    if(lang_group=="english"):
        print("Copying files...")
        shutil.copyfile(urls_csv,urls_csv_out)
        shutil.copyfile(emots_csv,emots_csv_out)
    
    else:
        df_urls = pd.read_csv(urls_csv)
        df_emots = pd.read_csv(emots_csv)

        anp_nanp = {}
        nanp_count = 0
        print("Creating indexes...")
        for i in range(df_urls.shape[0]):
            anp = df_urls.iloc[i,0]
            if not(anp in anp_nanp):
                anp_nanp[anp] = nanp_count
                nanp_count+=1
        
        print("Reindexing emotions...")
        for i in range(df_emots.shape[0]):
            df_emots.iloc[i,0] = lang_group+str(anp_nanp[df_emots.iloc[i,0]])

        print("Reindexing urls...")
        nanp_url_list = []
        nanp_url_list.append(("ANP","image_url"))
        for i in range(df_urls.shape[0]):
            nanp = lang_group+str(anp_nanp[df_urls.iloc[i,0]])
            url = df_urls.iloc[i,1]
            nanp_url_list.append((nanp,url))

        print("Writing csvs...")
        df_emots.to_csv(emots_csv_out, index=None, header=True)
        (pd.DataFrame(nanp_url_list)).to_csv(urls_csv_out, index=None, header=False)
    
def reindex_anp_lists_for_all():
    for lang_group in lang_groups:
        reindex_anp_lists_for_lang_group(lang_group)

def partit_anp_lists_for_lang_group(lang_group):

    if(lang_group=="english"):
        print("English will be ignored, returning")
        return
  
    print("Language group: "+str(lang_group))

    folder = image_dataset_dir+"MVSO/"
    urls_file = folder+"original/nanp_url/"+lang_group+".csv"
    parts_folder = folder+"original/nanp_url_parts/"+lang_group+"/"

    if(os.path.exists(parts_folder)):
        opt = input("Warning! Directory "+parts_folder+" already exists, \
if you clear it, the download indexes will be changed, clear it anyway? (y/n) ")
        if(opt=="y"):
            shutil.rmtree(parts_folder)
            os.makedirs(parts_folder)
    else:
        os.makedirs(parts_folder)

    df = pd.read_csv(urls_file)
    amount = df.shape[0]
    step = 1
    anp_url_list = []
    for i in range(0,amount,step):
        anp_url_list.append((i,df.iloc[i,0],df.iloc[i,1]))
        utils.show_progress("Preparing anps... ",i,amount)
    
    random.shuffle(anp_url_list)

    batch_size = min(1000,len(anp_url_list))

    k=0
    part_lists = []
    for i in range(len(anp_url_list)):
        if(k==0):
            part_list = []
        part_list.append(anp_url_list[i])
        k+=1
        if(k>=batch_size or i>=len(anp_url_list)-1):
            k=0
            part_lists.append(part_list)

    amount = len(part_lists)
    for i in range(len(part_lists)):
        utils.show_progress("Writing parts... ",i,amount)
        part_list = part_lists[i]
        fntmp1 = parts_folder+"urls_part"+str(i)+".csv"
        (pd.DataFrame(part_list)).to_csv(fntmp1, index=None, header=False)

    images_folder = folder+"images/"+lang_group+"/"
    if(os.path.exists(images_folder)):
        opt = input("Warning! Directory "+images_folder+" already exists, clear it? (y/n) ")
        if(opt=="y"):
            shutil.rmtree(images_folder)
            os.makedirs(images_folder)
    else:
        os.makedirs(images_folder)

def partit_anp_lists_for_all():
    for lang_group in lang_groups:
        partit_anp_lists_for_lang_group(lang_group)

def get_parts_to_download_for_lang_group(lang_group):

    if(lang_group=="english"):
        return 1000

    folder = image_dataset_dir+"MVSO/"
    nanp_parts_dir= folder+"/original/nanp_url_parts/"+lang_group+"/"

    count = 0
    for root, dirs, files in os.walk(nanp_parts_dir):
        for file in files:
            if file.endswith(".csv"):
                count+=1

    print("Parts for "+str(lang_group)+" group: 0 to "+str(count-1))
    
    return count

def view_down_completude(lang_group):

    os.system("clear")

    folder = image_dataset_dir+"MVSO/"

    n_parts = get_parts_to_download_for_lang_group(lang_group)

    vet = [0]*n_parts

    while(True):
        count=0
        amount=n_parts
        for i in range(0,n_parts):
            vet[i]=False
            if(os.path.exists(folder+"/images/"+lang_group+"/part"+str(i)+"/")):
                count+=1
                vet[i]=True
        percent = int(100*count/amount)
        
        i=0
        j=1
        strtmp1=""
        while(i<n_parts):
            if(vet[i]==True):
                strtmp1+="|"+"{:04d}".format(i)
            else:
                strtmp1+="|    "
            if(j>=40):
                j=0
                strtmp1+="|\n"
            i+=1
            j+=1
        
        utils.gotoxy(0,0)
        print("Created Folders: "+str(count)+"/"+str(amount)+" ("+str(percent)+"%)")
        print(strtmp1)
        time.sleep(1)

def download_anp_images_for_lang_group_part(lang_group,part_id):
    folder = image_dataset_dir+"MVSO/"
    urls_file = folder+"original/nanp_url_parts/"+lang_group+"/urls_part"+str(part_id)+".csv"
    if not(os.path.exists(urls_file)):
        print("Urls part file does not exist, returning!")
        return
    images_part_folder = folder+"images/"+lang_group+"/part"+str(part_id)+"/"
    if(os.path.exists(images_part_folder)):
        print("Warning: Overwriting part folder")
    else:
        os.makedirs(images_part_folder)
    df = pd.read_csv(urls_file)
    amount = df.shape[0]
    step = 1
    for i in range(0,amount,step):
        row = df.iloc[i,:].tolist()
        imgid,anp,url = row[0],row[1],row[2]
        save_filename = images_part_folder+anp+"_"+str(imgid)+url[-4:]
        utils.show_progress("Downloading "+lang_group+" part "+str(part_id)+": ",i,amount)
        if not(os.path.exists(save_filename)):
            try:
                urllib.request.urlretrieve(url, save_filename)
            except:
                pass

def download_anp_images_for_lang_group_part_interval(lang_group,part_id_i,part_id_f):
    if(lang_group=="english"):
        print("English will be ignored, returning")
        return

    for part_id in range(part_id_i,part_id_f+1):
        download_anp_images_for_lang_group_part(lang_group,part_id)

def prepare_labels_for_lang_group(lang_group):

    #ToDo: Stop using this global variable
    
    print("Language group: "+str(lang_group))

    print("  Preparing emotions...")

    folder = image_dataset_dir+"MVSO/"
    images_folder = folder+"images/"+lang_group+"/"
    emot_file = folder+"original/nanp_emot/ANP_emotion_mapping_"+lang_group+".csv"

    emot_names = []
    anp_names = []

    df_emot = pd.read_csv(emot_file,header=None)
    emot_names = df_emot.iloc[0,1:].tolist()
    
    anp_maxemot_dict = {}
    
    for i in range(1,df_emot.shape[0]):
        row = df_emot.iloc[i,:].tolist()
        anp = row[0]
        scores = row[1:]
        max_score = -1
        max_pos = -1
        for j in range(len(scores)):
            score = float(scores[j])
            if(score>max_score):
                max_score = score
                max_pos = j
        max_emot = emot_names[max_pos]   
        if(max_score>=anp_util_lim):
            anp_maxemot_dict[anp] = max_emot
        else:
            anp_maxemot_dict[anp] = "none"

    #4 Classes (swap  0 <--> 1)
    conv_labels_dict = {
        "ecstasy": 1,
        "joy": 1,
        "serenity": 3,
        "admiration": 3,
        "trust": 3,
        "acceptance": 3,
        "terror": 0,
        "fear": 0,
        "apprehension": 2,
        "amazement": 1,
        "surprise": 1,
        "distraction": 3,
        "grief": 2,
        "sadness": 2,
        "pensiveness": -1,
        "loathing": 0,
        "disgust": 2,
        "boredom": 2,
        "rage": 0,
        "anger": 0,
        "annoyance": 0,
        "vigilance": 2,
        "anticipation": 3,
        "interest": 3,
        "none": -1,
    }

    file_list = []
    for root, dirs, files in os.walk(images_folder):
        for file in files:
            if file.endswith(".jpg"):
                file_list.append(os.path.join(root, file))

    file_class_list = []
    file_class_list.append(("filename","class"))
    for f in file_list:
        i2 = -1
        while(f[i2]!="_"):
            i2-=1
        i1=i2
        while(f[i1]!="/"):
            i1-=1
        anp = f[i1+1:i2]
        i1-=1
        while(f[i1]!="/"):
            i1-=1
        fnam = "images/"+lang_group+"/"+f[i1+1:]
        emot = anp_maxemot_dict[anp]
        quad = conv_labels_dict[emot]
        if(quad!=-1):
            file_class_list.append((fnam,quad))

    print("  Writing csv...")

    fntmp1 = folder+"labels_"+lang_group+".csv"
    (pd.DataFrame(file_class_list)).to_csv(fntmp1, index=None, header=False)
    n_images = len(file_class_list)-1
    print("  Images: "+str(n_images))

    return fntmp1, n_images

def prepare_labels_for_all():
    folder = image_dataset_dir+"MVSO/"

    lang_labfiles = []

    n_images_t = 0

    for item in lang_groups:
        fntmp1, n_images = prepare_labels_for_lang_group(item)
        n_images_t+=n_images
        lang_labfiles.append(fntmp1)

    file_class_list = []
    file_class_list.append(("filename","class"))

    img_count = 0
    for i in range(len(lang_labfiles)):
        lbf = lang_labfiles[i]
        if(os.path.exists(lbf)):
            df = pd.read_csv(lbf)
            for j in range(df.shape[0]):
                img_count+=1
                utils.show_progress("Joining files... ",img_count,n_images_t)
                file_class_list.append((df.iloc[j,0],df.iloc[j,1]))
            os.remove(lbf)

    fntmp1 = folder+"labels_q.csv"
    (pd.DataFrame(file_class_list)).to_csv(fntmp1, index=None, header=False)
    print("Total images: "+str(len(file_class_list)-1))

def run():
    prepare_labels_for_all()
