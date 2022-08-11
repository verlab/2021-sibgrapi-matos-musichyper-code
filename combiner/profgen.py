from header import *

def group_and_plot_images(csv_filename):

    W = 1600+4
    H = 800+4
    w = 80
    h = 60
    n_x = 20
    n_y = 10
    wind_x = 100
    wind_y = 100

    df = pd.read_csv(csv_filename)
    classes = df.iloc[:,1]
    num_classes = max(classes)+1

    class_lists = []
    for i in range(num_classes):
        class_lists.append([])

    count = 0
    amount = 0
    for i in range(df.shape[0]):
        row = df.iloc[i,:].tolist()
        f,c = row[0],row[1]
        class_lists[c].append(f)
        amount+=1

    for i in range(num_classes):
        xi,yi = 0,0
        backg1 = np.zeros((H,W,3), np.uint8)
        backg1.fill(255)
        for j in range(len(class_lists[i])):
            utils.show_progress("Grouping images... ",count,amount)
            count+=1
            f = class_lists[i][j]
            img = cv2.imread(f)
            img = cv2.resize(img,(w,h), interpolation = cv2.INTER_AREA) 
            x_offset = 2+xi*w
            y_offset = 2+yi*h         
            backg1[y_offset:y_offset+img.shape[0], x_offset:x_offset+img.shape[1]] = img

            xi+=1
            if(xi>n_x-1):
                xi=0
                yi+=1
                if(yi>n_y-1):
                    yi=0  

        main_window = "Class "+str(i)
        if(False):
            cv2.namedWindow(main_window)
            cv2.moveWindow(main_window,wind_x,wind_y)
            cv2.imshow(main_window,backg1)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            cv2.imwrite(out_dir+main_window+".jpg",backg1)

def draw_point(ax,x,y,color,marker):
    x = (x*(+0.78))*400+400 #0.85
    y = (y*(-0.78))*400+400 #0.85
    ax.plot(x,y,color=color,marker=marker)

def generate_plots(i,smt_pnts,vsd1,asd1,vsc1,asc1,frm1,stp1,outdir,dtp1,tit1,imb1):

    import matplotlib
    import matplotlib.pyplot as plt

    i1,i2,k,n = i[0],i[1],i[2],i[3]

    print("  Init part "+str(k)+"/"+str(n))

    out_imgs = []

    fig = plt.figure(figsize=(10, 5))
    fig.suptitle(tit1, fontsize=16)

    grid = plt.GridSpec(4,2, wspace=0.2, hspace=0.1)

    ax1 = fig.add_subplot(grid[0:,1])
    ax2 = fig.add_subplot(grid[0:2,0])
    ax3 = fig.add_subplot(grid[2,0])
    ax4 = fig.add_subplot(grid[3,0])

    ax_list = [ax1,ax2,ax3,ax4]

    for i in range(i1,i2):

        item = smt_pnts[i]

        show_emojis = True
        shadow_level = 1
        if(show_emojis==True):
            
            ax1.imshow(imb1)
            ax1.set_ylabel("Emotion")

            strtmp1 = utils.time2minsec(item[0])
            
            if(shadow_level>=2):
                amttmp1=len(smt_pnts)
                for j in range(max(i-amttmp1,0),i):
                    item_j = smt_pnts[j]
                    draw_point(ax1,item_j[1],item_j[2],(0,0,0.5,0.02) ,"o")

            if(shadow_level>=1):
                amttmp1=5
                alptmp1=0
                for j in range(max(i-amttmp1,0),i):
                    alptmp1+=1/amttmp1
                    item_j = smt_pnts[j]
                    draw_point(ax1,item_j[1],item_j[2],(0,0,1,min(alptmp1,1)) ,"o")
            
            draw_point(ax1,item[3],item[4],"silver","x")
            draw_point(ax1,item[1],item[2],'m',"o")

        for item in ax_list:
            item.set_xticks([],[])
            item.set_yticks([],[])
            item.set_xticklabels([])
            item.set_yticklabels([])

        if(dtp1=="audio"):
            ax2.set_ylabel("Song Wave")
            ax2.plot(frm1)
            ax2.axvline(i*stp1, c='m')
        
        if(dtp1=="video" or dtp1=="image"):
            ax2.set_ylabel("Video Frame")
            frm1 = plt.imread(smt_pnts[i][5])
            ax2.imshow(frm1)

        ax3.set_ylabel("Valence")
        ax3.set_ylim([-1, +1])
        ax3.plot(vsd1,color='silver')
        ax3.plot(vsc1,color='blue')
        ax3.axvline(i, c='m')

        ax4.set_ylabel("Arousal")
        ax4.set_ylim([-1, +1])
        ax4.plot(asd1,color='silver')
        ax4.plot(asc1,color='blue')
        ax4.axvline(i, c='m')

        #plt.savefig(outdir+str(i)+".png")

        out_imgs.append(utils.plt2array(fig))

        for item in ax_list:
            item.clear()
        plt.close()

    print("  Done part "+str(k)+"/"+str(n))

    return out_imgs

class profgen():
    def __init__(self,data_type,num_classes,fnam_orig,rend_vid,use_gpu):

        import librosa

        print("Initializing profgen for "+data_type+"...")

        self.data_type = data_type
        self.num_classes = num_classes
        self.render_video = rend_vid
        self.anime_mode = False
        self.video_images = None

        if(use_gpu==True):
            use_gpu = True if(torch.cuda.is_available()) else False
        self.use_gpu = use_gpu

        cache_dir_sub = cache_dir+self.data_type+"/"
        if not(os.path.exists(cache_dir_sub)):
            os.mkdir(cache_dir_sub)

        i=-1
        while(fnam_orig[i]!="/"):
            i-=1   
        fnam_copy = cache_dir_sub+fnam_orig[i+1:]

        self.input_filename_orig = fnam_orig
        self.input_filename_copy = fnam_copy

        self.main_dir = self.input_filename_copy[:-4]+"/"
        if not(os.path.exists(self.main_dir)):
            os.mkdir(self.main_dir)

        self.frames_dir = self.main_dir+"frames/"
        if not(os.path.exists(self.frames_dir)):
            os.mkdir(self.frames_dir)

        self.plots_dir = self.main_dir+"plots/"
        if(os.path.exists(self.plots_dir)):
            shutil.rmtree(self.plots_dir)
        os.mkdir(self.plots_dir)

        self.avi_filename = self.main_dir+"video.avi"
        self.mp4_filename = self.input_filename_copy[:-4]+".mp4"
        self.png_filename = self.input_filename_copy[:-4]+".png"
        
        self.render_step = 1

        if(data_type=="audio"):
            self.song_lib_y, self.song_lib_sr = librosa.load(self.input_filename_orig)
            self.song_lib_len = int(librosa.get_duration(y=self.song_lib_y,sr=self.song_lib_sr))

            self.start_offset = 2
            self.smot_gen_n = 15 #Não alterar!
            self.lab_get_intv = 500 #Não alterar!
            self.accel_ref = 0.0003

            if(self.render_video==True):
                self.anime_mode = True
                self.smot_gen_n = 3
                self.accel_ref = 0.005

        elif(data_type=="video"):
            self.video_cv2_cap = cv2.VideoCapture(self.input_filename_orig)
            self.video_cv2_len = int(cv2.VideoCapture.get(self.video_cv2_cap, int(cv2.CAP_PROP_FRAME_COUNT)))
            self.video_cv2_fps = self.video_cv2_cap.get(cv2.CAP_PROP_FPS)
            self.video_ffnam_list = []
            
            self.start_offset = 0
            self.smot_gen_n = 3 #Não alterar!
            self.lab_get_intv = 100 #Não alterar!
            self.accel_ref = 0.003

        elif(data_type=="image"):
            
            self.video_ffnam_list = []

            self.start_offset = 0
            self.smot_gen_n = 3
            self.lab_get_intv = 3000
            self.accel_ref = 0.03

        self.points_csv = self.main_dir+"labels.csv"
        if not(os.path.exists(self.points_csv) and self.anime_mode==False):
            print("Copying "+fnam_orig+" to cache... ")
            shutil.copyfile(fnam_orig,fnam_copy)

        self.num_classes_sqrt = int(math.sqrt((self.num_classes)))
        self.image_amount = 0
        self.labels = None
        self.smoothed_points = None

        #ToDo: Improve the accel equation to be relative to lab_get_intv

        accel = self.accel_ref/((1000/self.lab_get_intv)/self.smot_gen_n)
        spdmx = accel*10

        self.hspeed=0
        self.vspeed=0
        self.haccel=accel
        self.vaccel=accel
        self.hspeed_max=spdmx
        self.vspeed_max=spdmx

        print("Loading model...")
        
        if(self.data_type=="audio"):

            self.audio_labelmode = "axis"
            self.audio_netname = "mernet01"
            self.audio_modeldir = saved_models_dir+"audio_main/"

            self.audio_modsttdic_val = self.audio_modeldir+self.audio_netname+"_v.pth"
            self.audio_modsttdic_ars = self.audio_modeldir+self.audio_netname+"_a.pth"
            self.audio_modsttdic_qud = self.audio_modeldir+self.audio_netname+"_q.pth"

            audio_ncf = self.num_classes if(self.audio_labelmode=="quds") else self.num_classes_sqrt
            self.audio_model = learning.models.choose_model(self.audio_netname,audio_ncf,self.use_gpu)

        if(self.data_type=="video" or self.data_type=="image"):

            self.video_labelmode = "quds"
            self.video_netname = "resnet50ext"
            self.video_modeldir = saved_models_dir+"video_main/"

            self.video_modsttdic_val = self.video_modeldir+self.video_netname+"_v.pth"
            self.video_modsttdic_ars = self.video_modeldir+self.video_netname+"_a.pth"
            self.video_modsttdic_qud = self.video_modeldir+self.video_netname+"_q.pth"

            video_ncf = self.num_classes if(self.video_labelmode=="quds") else self.num_classes_sqrt
            self.video_model = learning.models.choose_model(self.video_netname,video_ncf,self.use_gpu)

    def get_valars_from_quads(self,quadrants,num_classes_sqrt):

        valences = []
        arousals = []
        nc_ax = num_classes_sqrt
        for quadrant in quadrants:
            v,a,q = 0,0,0
            while(q<quadrant):
                q+=1
                v+=1
                if(v>(nc_ax-1)):
                    v=0
                    a+=1
            valences.append(v)
            arousals.append(a)

        return valences, arousals

    def get_valars_from_quads_vid4q_aux(self,quadrants,num_classes_sqrt):
        valences = []
        arousals = []
        for q in quadrants:
            if(q==0):
                v,a=1,1
            elif(q==1):
                v,a=0,1
            elif(q==2):
                v,a=0,0
            elif(q==3):
                v,a=1,0
            valences.append(v)
            arousals.append(a)

        return valences, arousals

    def move_towards_point(self,x,y,xf,yf):

        if(self.data_type=="image"):
            return (xf,yf)

        if(x<xf-self.hspeed):
            self.hspeed+=self.haccel
        elif(x>xf+self.hspeed):
            self.hspeed-=self.haccel
        else:
            self.hspeed=0
        
        if(y<yf-self.hspeed):
            self.vspeed+=self.vaccel
        elif(y>yf+self.hspeed):
            self.vspeed-=self.vaccel
        else:
            self.vspeed=0

        if(self.hspeed>+self.hspeed_max):
            self.hspeed=+self.hspeed_max
        if(self.hspeed<-self.hspeed_max):
            self.hspeed=-self.hspeed_max

        if(self.vspeed>+self.vspeed_max):
            self.vspeed=+self.vspeed_max
        if(self.vspeed<-self.vspeed_max):
            self.vspeed=-self.vspeed_max

        if(abs(x-xf)<=abs(self.hspeed*2)):
            self.hspeed=0
        
        if(abs(y-yf)<=abs(self.vspeed*2)):
            self.vspeed=0

        x+=self.hspeed
        y+=self.vspeed

        x=min(max(x,-1),+1)
        y=min(max(y,-1),+1)

        return(x,y)

    def generate_song_labels(self):

        y, sr, sl = self.song_lib_y, self.song_lib_sr, self.song_lib_len

        pieces_list = []
        for time_id in range(0,sl*1000,self.lab_get_intv):
            filename_part = self.input_filename_copy+"-"+str(time_id)
            pieces_list.append((filename_part))
        
        if(self.audio_labelmode=="quds"):
            quadrants = learning.trainer.predict_list(self.audio_model,self.audio_modsttdic_qud,pieces_list,"audio", self.use_gpu)
            valences, arousals = self.get_valars_from_quads(quadrants,self.num_classes_sqrt)

        elif(self.audio_labelmode=="axis"):
            valences = learning.trainer.predict_list(self.audio_model,self.audio_modsttdic_val,pieces_list,"audio", self.use_gpu)
            arousals = learning.trainer.predict_list(self.audio_model,self.audio_modsttdic_ars,pieces_list,"audio", self.use_gpu)
        
        labels = []
        for i in range(min(len(valences),len(arousals))):
            time_id = int(i/(1000/self.lab_get_intv))
            if(i>self.start_offset):
                labels.append((time_id,valences[i],arousals[i]))
            else:
                labels.append((time_id,valences[self.start_offset],arousals[self.start_offset]))

        self.labels = labels

        for root, dirs, files in os.walk(self.frames_dir):
            for file in files:
                if file.endswith(".wav"):
                    fnam = os.path.join(root, file)
                    if(os.path.exists(fnam)):
                        os.remove(fnam)

    def generate_video_labels(self):

        plots_dir = self.plots_dir

        images_list = []

        extract_frames = True
        if(os.path.exists(self.frames_dir+"0.jpg")):
            extract_frames = False

        count=0
        amount = self.video_cv2_len
        vc = self.video_cv2_cap
        rd_intv = self.smot_gen_n

        if(extract_frames):
            while(True):
                utils.show_progress("  Extracting video frames... ",count,amount)
                filename_frame = self.frames_dir+str(count)+".jpg"
                count+=1
                success,frame = vc.read()
                if(success==False or count>amount):
                    break
                else:
                    cv2.imwrite(filename_frame,cv2.resize(frame,global_out_size))
        else:
            print("  Note: Using cache extracted frames!")

        for i in range(amount):
            filename_frame = self.frames_dir+str(i)+".jpg"
            if(os.path.exists(filename_frame)):
                self.video_ffnam_list.append(filename_frame)
                if(i%rd_intv==0):
                    images_list.append((filename_frame))
            
        if(self.video_labelmode=="quds"):
            quadrants = learning.trainer.predict_list(self.video_model,self.video_modsttdic_qud,images_list,"video", self.use_gpu)
            valences, arousals = self.get_valars_from_quads_vid4q_aux(quadrants,self.num_classes_sqrt)
        elif(self.video_labelmode=="axis"):
            print("Not implemented yet!") #ToDo: Implement this code

        labels = []
        ttmp1=0
        time_id=0
        for i in range(min(len(valences),len(arousals))):
            ttmp1+=1
            if(ttmp1>=self.video_cv2_fps):
                ttmp1=0
                time_id+=1
            labels.append((time_id,valences[i],arousals[i]))

        self.labels = labels

    def generate_image_labels(self):

        images_list = []
        df = pd.read_csv(self.input_filename_copy)
        for i in range(df.shape[0]):
            filename_frame = df.iloc[i,0]
            images_list.append(filename_frame)
            self.video_ffnam_list.append(filename_frame)

        if(self.video_labelmode=="quds"):
            quadrants = learning.trainer.predict_list(self.video_model,self.video_modsttdic_qud,images_list,"video", self.use_gpu)
            valences, arousals = self.get_valars_from_quads(quadrants,self.num_classes_sqrt)
        elif(self.video_labelmode=="axis"):
            print("Not implemented yet!") #ToDo: Implement this code

        labels = []
        for i in range(min(len(valences),len(arousals))):
            time_id = "0"
            labels.append((time_id,valences[i],arousals[i]))
        
        self.labels = labels

        lbtmp1 = []
        for i in range(len(quadrants)):
            lbtmp1.append((images_list[i],quadrants[i]))
        fntmp1 = cache_dir+"temp.csv"
        (pd.DataFrame(lbtmp1)).to_csv(fntmp1, index=None, header=False)
        group_and_plot_images(fntmp1)
        exit()

    def generate_smoothed_points(self):

        print("Generating points for file "+self.input_filename_copy)

        if(os.path.exists(self.points_csv) and self.anime_mode==False):
            print("  Note: Using cache generated points")
            smt_pnts = []
            df = pd.read_csv(self.points_csv, header=None)
            for i in range(df.shape[0]):
                smt_pnt=[]
                for j in range(df.shape[1]):
                    smt_pnt.append(df.iloc[i,j])
                smt_pnts.append(smt_pnt)
            
            self.smoothed_points = smt_pnts

        else:

            if(self.data_type=="audio"):
                self.generate_song_labels()
            elif(self.data_type=="video"):
                self.generate_video_labels()
            elif(self.data_type=="image"):
                self.generate_image_labels()

            labels = self.labels

            tav_pnts = []
            arousals = []
            valences = []
            times = []
            
            for item in labels:
                time_id = int(item[0])
                time_val = int(item[1])
                time_ars = int(item[2])

                nc_ax = self.num_classes_sqrt
                labels_ref = [0]*nc_ax
                inc = 2/nc_ax
                lb_min = -1+inc/2
                v = lb_min
                k = 0

                while(k<nc_ax):
                    labels_ref[k] = v
                    k+=1
                    v+=inc

                valence = labels_ref[time_val]
                arousal = labels_ref[time_ars]

                if(time_id>=self.start_offset):
                    tav_pnts.append((time_id-self.start_offset,valence,arousal))
                    times.append(time_id)
                    valences.append(valence)
                    arousals.append(arousal)  

            (x,y) = (0,0)
            smt_pnts = []
            for tav_p in tav_pnts:
                xf = tav_p[1]
                yf = tav_p[2]
                for i in range(self.smot_gen_n):
                    (x,y) = self.move_towards_point(x,y,xf,yf)
                    smt_pnts.append((tav_p[0],x,y,xf,yf))

            if(self.data_type=="video"):
                smt_pnts2 = []
                for i in range(len(self.video_ffnam_list)):
                    smt_pnts2.append((
                        smt_pnts[i][0],
                        smt_pnts[i][1],
                        smt_pnts[i][2],
                        smt_pnts[i][3],
                        smt_pnts[i][4],
                        self.video_ffnam_list[i]
                    ))
                smt_pnts = smt_pnts2

            self.smoothed_points = smt_pnts

            if(self.anime_mode==False):
                (pd.DataFrame(self.smoothed_points)).to_csv(self.points_csv, index=None, header=False)
        
    def generate_emot_images(self):

        smt_pnts = self.smoothed_points
        labels = self.labels

        imd1 = self.plots_dir
        dtp1 = self.data_type

        vsd1, asd1, vsc1, asc1 = [],[],[],[]

        for j in range(len(smt_pnts)):
            if(j>self.start_offset*self.smot_gen_n):
                vsc1.append(smt_pnts[j][1])
                asc1.append(smt_pnts[j][2])
            vsd1.append(smt_pnts[j][3])
            asd1.append(smt_pnts[j][4])

        if(self.data_type=="audio"):
            y, sr = self.song_lib_y, self.song_lib_sr
            stp1 = (sr/self.smot_gen_n)/(1000/self.lab_get_intv)/100
            frm1 = []
            for i in range(0,len(y[:-self.start_offset*sr]),100):
                frm1.append(y[i])
 
        if(self.data_type=="video" or self.data_type=="image"):
            frm1 = self.video_ffnam_list
            stp1 = -1
        
        tit1 = self.input_filename_copy
        k=-1
        while(tit1[k]!="/"):
            k-=1
        tit1 = tit1[k+1:]

        print("Creating animation...")

        amount = len(smt_pnts)
        i_list = [i for i in range(0,amount,self.render_step)]
        video_images = []
        imb1 = plt.imread("assets/img_back1.png")

        njobs=num_cores
        ljobs = utils.get_ljobs(0,len(i_list),njobs)
        if(self.render_video==False):
            i_list = [amount-1]
            ljobs = [[amount-1,amount,1,1]]
        louts = Parallel(n_jobs=njobs)(
            delayed(generate_plots)(i,smt_pnts,vsd1,asd1,vsc1,asc1,frm1,stp1,imd1,dtp1,tit1,imb1)
                for i in ljobs)
        for lout in louts:
            for item in lout:
                video_images.append(item)

        self.image_amount = len(i_list)
        self.video_images = video_images

    def make_video_from_images(self):

        #if(self.render_video==False):
        #    for root, dirs, files in os.walk(self.plots_dir):
        #        for file in files:
        #            if file.endswith(".png"):
        #                fntmp1 = os.path.join(root, file)
        #                fntmp2 = self.png_filename
        #                shutil.copyfile(fntmp1,fntmp2)
        #                fntmp3 = utils.copy_file_to_out_dir(fntmp2)
        #                print("Done!! Out file: "+str(fntmp3))
        #                return

        if(self.render_video==False):
            cv2.imwrite(self.png_filename,self.video_images[0])
            fntmp2 = self.png_filename
            shutil.copyfile(fntmp1,fntmp2)
            fntmp3 = utils.copy_file_to_out_dir(fntmp2)
            print("Done!! Out file: "+str(fntmp3))
            return

        import ffmpeg
        
        images = []
        for i in range(0,self.image_amount*self.render_step,self.render_step):
            for j in range(self.render_step):
                #images.append(str(i)+".png")
                images.append(self.video_images[i])
        
        #frame = cv2.imread(os.path.join(self.plots_dir, images[0]))
        frame = self.video_images[0]

        height, width, layers = frame.shape
        fps = int(self.smot_gen_n*1000/self.lab_get_intv)
        video = cv2.VideoWriter(self.avi_filename, 0, fps, (width,height))
        for i in range(len(images)):
            utils.show_progress("Rendering video... ",i,len(images))
            #video.write(cv2.imread(os.path.join(self.plots_dir, images[i])))
            video.write(self.video_images[i])

        cv2.destroyAllWindows()
        video.release()

        if(os.path.exists(self.mp4_filename)):
            os.remove(self.mp4_filename)

        print("Inserting Audio... ")

        if(self.data_type=="audio"):  
            inp_video = ffmpeg.input(self.avi_filename)
            inp_audio = ffmpeg.input(self.input_filename_orig)          
            ffmpeg.concat(inp_video, inp_audio, v=1, a=1).output(self.mp4_filename).run()

        if(self.data_type=="video" or self.data_type=="image"):
            os.system("ffmpeg -i "+str(self.avi_filename)+" -strict -2 "+str(self.mp4_filename))

        mp4_filename_f = utils.copy_file_to_out_dir(self.mp4_filename)            

        print("Done!! Out file: "+str(mp4_filename_f))

    def clear_temp_files(self):
        if(os.path.exists(self.input_filename_copy)):
            os.remove(self.input_filename_copy)
        if(os.path.exists(self.avi_filename)):
            os.remove(self.avi_filename)
        if(os.path.exists(self.mp4_filename)):
            os.remove(self.mp4_filename)
        if(os.path.exists(self.png_filename)):
            os.remove(self.png_filename)

    def plot_emotion_profile(self,rend_vid):
        self.render_video = rend_vid
        self.generate_smoothed_points()      
        self.generate_emot_images()
        self.make_video_from_images()
        self.clear_temp_files()

    def get_emotion_profile(self):
        self.generate_smoothed_points()
        self.clear_temp_files()
        return self.smoothed_points