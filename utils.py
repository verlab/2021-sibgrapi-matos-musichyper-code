from header import *

def install_dependencies():
    option = input("Install all dependencies with pip3? (y/n)")
    if(option=="y"):
        os.system('sudo apt update')
        os.system('sudo apt install python3-pip')
        os.system('pip3 install numpy')
        os.system('pip3 install pandas')
        os.system('pip3 install matplotlib')
        os.system('pip3 install librosa')
        os.system('pip3 install essentia')
        os.system('pip3 install ffmpeg')
        os.system('pip3 install opencv-python')
        os.system('pip3 install torch')
        os.system('pip3 install torchvision')
        os.system('pip3 install importlib')
        os.system('pip3 install dijkstra')
        os.system('pip3 install dtw')
    else:
        print("Install aborted!")

def wait(message="Press enter!"):
    input(message)

def clear_pycache():
    os.system('find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf')

def point_distance(x1,y1,x2,y2):
    return math.sqrt((x2-x1)**2+(y2-y1)**2)

def time2minsec(value):
    minutes = int(value/60)
    seconds = value%60
    str1 = "0" if(minutes<10) else ""
    str2 = "0" if(seconds<10) else ""
    strf = str1+str(minutes)+":"+str2+str(seconds)
    return strf

def minsec2time(value):
    strf = int(value[0:2])*60 + int(value[3:5])
    return strf

def max_position(vect):
    max_pos = 0
    for i in range(len(vect)):
        if(vect[i]>vect[max_pos]):
            max_pos = i
    return max_pos
    
def save_log(log_str,log_filename):
    print(log_str)
    try:
        with open(log_filename,'a') as f:
            f.write("\n")
            f.write(log_str)
    except:
        print("Cannot save log!")

def show_progress(message,count,amount):
    amount = max(1,amount-1)
    percent = int(100*count/amount)
    interval = max(1,int(amount/100))
    if(percent>98):
        interval = 1
    if(count%interval==0):
        if(count!=0):
            sys.stdout.write("\033[F")
        print(message+str(min(count,amount))+"/"+str(amount)+" ("+str(percent)+"%)"+"                ")

def git_clean():
    os.system('git checkout --orphan newBranch')
    os.system('git add -A')
    os.system('git commit -am "Initial commit"')
    os.system('git branch -D master')
    os.system('git branch -m master')
    os.system('git push -f origin master')

def git_push(message):
    os.system('find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf')
    os.system('git add -A')
    os.system('git commit -a -m "'+str(message)+'"')
    os.system('git push')

def git_pull():
    os.system('git pull')

def gotoxy(x,y):
    print ("%c[%d;%df" % (0x1B, y, x), end='')

def count_wave_parts():
    while(True):
        count=0
        for root, dirs, files in os.walk("/srv/storage/datasets/Diognei/Audio/DEAM/songs/"):
            for file in files:
                if file.endswith(".wav"):
                    count+=1
        os.system("clear")
        print(count)
        time.sleep(3)

def check_audio_samples(audio_list):
    df = pd.read_csv(audio_list)
    for i in range(df.shape[0]):
        print(str(i)+"/"+str(df.shape[0]))
        filename = df.iloc[i,0]
        j=-1
        while(filename[j]!='-'):
            j-=1
        filename = filename[:j-4]+"/"+filename[j+1:]+".csv"
        df_feat = pd.read_csv(filename)
        features = df_feat.iloc[:,0]
        if(len(features)!=48):
            print(filename)
            utils.wait()

def str2bool(s):
    b = True if(s in ["true","True","1","y","Y"]) else False 
    return b

def copy_file_to_temp_dir(filename):
    i=-1
    while(filename[i]!="/"):
        i-=1   
    filename_cp = cache_dir+filename[i+1:]
    shutil.copyfile(filename,filename_cp)
    return filename_cp

def copy_file_to_out_dir(filename):
    i=-1
    while(filename[i]!="/"):
        i-=1   
    filename_cp = out_dir+filename[i+1:]
    shutil.copyfile(filename,filename_cp)
    return filename_cp

def copy_file_to_any_dir(filename,dir):
    i=-1
    while(filename[i]!="/"):
        i-=1   
    filename_cp = dir+filename[i+1:]
    shutil.copyfile(filename,filename_cp)
    return filename_cp

def save_model_as_main(version_name):

    model_dir = models_dir+"/"+version_name+"/"

    if(os.path.exists(model_dir)):

        config_file = model_dir+"config.json"
        config_dict = json.load(open(config_file))
        
        model_name = config_dict["model_name"]
        dataset_name = config_dict["dataset_name"]
        labels_suffix = config_dict["labels_suffix"]

        labels_suffix = "_"+labels_suffix[0]

        if(dataset_name=="DEAM"):
            data_type = "audio"
        else:
            data_type = "video"

        in_fnam = models_dir+version_name+"/model.pth"
        out_dir = saved_models_dir+data_type+"_main/"
        out_fnam = out_dir+model_name+labels_suffix+".pth"

        opt = "y"
        if(os.path.exists(out_fnam)):
            opt = input("Overwrite file "+str(out_fnam)+" ? (y/n) ")

        if(opt=="y"):
            shutil.copyfile(in_fnam,out_fnam)
            print("Copied "+in_fnam+" to "+out_fnam)
    else:
        print("Model dir not found!")
        return
        
def save_matrix_with_labels(matrix,x_labels,y_labels,filename):

    import matplotlib
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    im = ax.imshow(matrix)

    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            text = ax.text(j, i, matrix[i, j],
                        ha="center", va="center", color="w")

    fig.tight_layout()
    plt.show()
    plt.savefig(filename)

def plot_matrix(mat,show_value=False):
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(1,1,1)
    ax1.matshow(mat,cmap=plt.cm.Blues)
    ax1.set_xticks([],[])
    ax1.set_yticks([],[])
    if(show_value==True):
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax1.text(j, i, str(int(mat[i,j])), va='center', ha='center')
    plt.show()

def path2fnam(fnam_path):
    i=-1
    while(fnam_path[i-1]!="/"):
        i-=1
    fnam_only = fnam_path[i:]
    return fnam_only

def get_ljobs(ini,size,njobs):
    ljobs = []
    njobs = min(njobs,size)
    div=max(1,(size-ini)/njobs)
    i1=ini
    i2=i1+div

    for k in range(njobs):
        if(k==njobs-1):
            i2=size
        ljobs.append([int(i1),int(i2),k+1])
        i1+=div
        i2+=div
    
    for i in range(len(ljobs)):
        ljobs[i].append(k+1)

    return ljobs

def convert_videos():
    videos_dir = "/srv/storage/datasets/Diognei/Video/MSHP/"
    vcache_dir = "/srv/storage/datasets/Diognei/Cache/video/"
    output_dir = "/srv/storage/datasets/Diognei/Cache/temp/"
    file_list = []
    for root, dirs, files in os.walk(videos_dir):
        for file in files:
            if file.endswith(".mp4"):
                vc_fnam = vcache_dir+file[:-4]+"/frames/"
                file_list.append(vc_fnam)

    for k in range(len(file_list)):

        vidf_dir = file_list[k]

        print("Current dir: "+vidf_dir+" ("+str(k+1)+"/"+str(len(file_list))+") ")
        
        i = -9
        while(vidf_dir[i]!="/"):
            i-=1
        
        avi_filename = output_dir+vidf_dir[i+1:-8]+".avi"
        mp4_filename = output_dir+vidf_dir[i+1:-8]+".mp4"

        if not(os.path.exists(mp4_filename)):
            images = []
            i = 0
            while(True):
                fnam = vidf_dir+str(i)+".jpg"
                if(os.path.exists(fnam)):
                    images.append(fnam)
                else:
                    break
                i+=1

            video = cv2.VideoWriter(avi_filename, 0, 30, (640,480))
            for i in range(len(images)):
                utils.show_progress("  Rendering video... ",i,len(images))
                video.write(cv2.imread(images[i]))

            cv2.destroyAllWindows()
            video.release()

            os.system("ffmpeg -i "+str(avi_filename)+" -strict -2 "+str(mp4_filename))

            os.remove(avi_filename)
        else:
            print("Output file already exists")

    print("All Done!")

def plt2array(fig):
    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
            
def compress_video(filename):
    filename_c = filename[:-4] +"_cmp.mp4"
    if(os.path.exists(filename_c)):
        os.remove(filename_c)
    os.system("ffmpeg -i " + filename +" -b 500k " + filename_c)
    return filename_c
