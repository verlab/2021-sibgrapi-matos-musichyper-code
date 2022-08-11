from header import *

def validate_pair(song_points,video_points):
    if(len(song_points)>len(video_points)):
        print("  Song is bigger than video, ignoring it.")
        return False
    elif(len(song_points)<600):
        print("  Song is less than 20 seconds, ignoring it.")
        return False
    elif(len(song_points)>6500):
        print("  Ignoring song with >6500 frames (temp).")
        return False
    else:
        return True

def preprocess_points(point_list,type):

    normalize=False
    if(normalize==False):
        return point_list

    #Normalize points:

    p1min = +10
    p1max = -10
    p2min = +10
    p2max = -10

    for p in point_list:
        if(p[1]<p1min):
            p1min = p[1]
        if(p[1]>p1max):
            p1max = p[1]
        if(p[2]<p2min):
            p2min = p[2]
        if(p[2]>p2max):
            p2max = p[2]

    p1rng = p1max-p1min
    p2rng = p2max-p2min

    point_list_f = []

    for p in point_list:
        if(p1rng!=0):
            p1nrm = min(+1,max(-1,(p[1]-(p1max-p1rng/2))*(2/p1rng)))
        if(p2rng!=0):
            p2nrm = min(+1,max(-1,(p[2]-(p2max-p2rng/2))*(2/p2rng)))

        if(len(point_list[0])==5):
            point_list_f.append((p[0],p1nrm,p2nrm,p[3],p[4]))
        else:
            point_list_f.append((p[0],p1nrm,p2nrm,p[3],p[4],p[5]))

    return point_list_f

def draw_point(ax,x,y,color,marker):
    x = (x*(+0.78))*400+400
    y = (y*(-0.78))*400+400
    ax.plot(x,y,color=color,marker=marker)

def generate_plots(i,ftit1,imb1,d_lists,plots_dir):

    i1,i2,k,n = i[0],i[1],i[2],i[3]

    out_imgs = []

    print("  Init part "+str(k)+"/"+str(n))

    v_imgs = d_lists[0]
    s_vals = d_lists[1]
    v_vals = d_lists[2]
    s_arss = d_lists[3]       
    v_arss = d_lists[4]
    p_sims = d_lists[5]

    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(ftit1, fontsize=16)

    grid = plt.GridSpec(7,9, wspace=0.8, hspace=0.2)

    ax1 = fig.add_subplot(grid[0:7,0:6])
    ax2 = fig.add_subplot(grid[0:4,6:9])
    ax3 = fig.add_subplot(grid[4:5,6:9])
    ax4 = fig.add_subplot(grid[5:6,6:9])
    ax5 = fig.add_subplot(grid[6:7,6:9])

    ax_list = [ax1,ax2,ax3,ax4,ax5]

    for i in range(i1,i2):

        limtmp1 = 1.2
        m_sim = round(np.mean(p_sims),global_round_numbs)
        imv1 = plt.imread(v_imgs[i])

        ax1.set_title("Video")
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.imshow(imv1)

        ax2.set_title("Emotion (blue=video, green=audio)")
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.imshow(imb1)

        amttmp1=5
        alptmp1=0
        for j in range(max(i-amttmp1,0),i):
            alptmp1+=1/amttmp1
            draw_point(ax2,s_vals[j],s_arss[j],(0,1,0,min(alptmp1,1)),"o")
            draw_point(ax2,v_vals[j],v_arss[j],(0,0,1,min(alptmp1,1)),"o")

        draw_point(ax2,s_vals[i],s_arss[i],'green',"o")
        draw_point(ax2,v_vals[i],v_arss[i],'blue',"o")

        ax3.set_ylabel("Valence (X)")
        ax3.set_xticks([])
        ax3.set_ylim([-limtmp1, +limtmp1])
        ax3.plot(s_vals,label="audio",color="green")
        ax3.plot(v_vals,label="video",color="blue")
        ax3.axvline(i, c='m')
        
        ax4.set_ylabel("Arousal (Y)")
        ax4.set_xticks([])
        ax4.set_ylim([-limtmp1, +limtmp1])
        ax4.plot(s_arss,label="audio",color="green")
        ax4.plot(v_arss,label="video",color="blue")
        ax4.axvline(i, c='m')

        ax5.set_xlabel("Frame id")
        ax5.set_ylabel("Similarity")
        ax5.set_ylim([-0.2, +limtmp1])
        ax5.plot(p_sims,label="m: "+str(m_sim),color="red")
        ax5.legend(loc='lower right') 
        ax5.axvline(i, c='m')

        #img_fnam = plots_dir+str(i)+".png"
        #plt.savefig(img_fnam)
        #out_imgs.append(img_fnam)
        out_imgs.append(utils.plt2array(fig))

        for item in ax_list:
            item.clear()
        plt.close()

    print("  Done part "+str(k)+"/"+str(n))

    return out_imgs

def render_video(song_fnam,song_points,video_fnam,video_points,opt_method,save_dir,animation):
    
    vtit1 = video_fnam
    k=-1
    while(vtit1[k]!="/"):
        k-=1
    vtit1 = vtit1[k+1:-4]

    stit1 = song_fnam
    k=-1
    while(stit1[k]!="/"):
        k-=1
    stit1 = stit1[k+1:-4]

    mtit1 = opt_method

    ftit1 = vtit1+"_"+stit1+"_"+mtit1

    cache_dir_sub = cache_dir+"combiner/"
    if not(os.path.exists(cache_dir_sub)):
        os.mkdir(cache_dir_sub)
    redut_dir_video = cache_dir_sub+vtit1+"/"
    if not(os.path.exists(redut_dir_video)):
        os.mkdir(redut_dir_video)
    redut_dir_audio = redut_dir_video+stit1+"/"
    if not(os.path.exists(redut_dir_audio)):
        os.mkdir(redut_dir_audio)
    redut_dir_method = redut_dir_audio+mtit1+"/"
    if not(os.path.exists(redut_dir_method)):
        os.mkdir(redut_dir_method)

    redut_dir = redut_dir_method
    plots_dir = redut_dir+"plots/"
    if not(os.path.exists(plots_dir)):
        os.mkdir(plots_dir)

    import ffmpeg
    import librosa

    fps = global_out_fps

    song_fnam_cut = redut_dir+"song_piece.wav"
    scut_len = len(song_points)/fps
    y, sr = librosa.load(song_fnam,offset=0,duration=scut_len,mono=False)
    librosa.output.write_wav(song_fnam_cut,y,sr)

    video_images = []
    if(animation==True):

        print("Rendering animation... ")

        v_imgs = []
        s_vals = []
        v_vals = []
        s_arss = []        
        v_arss = []
        p_sims = []

        for j in range(min(len(song_points),len(video_points))):
            song_point = song_points[j]
            video_point = video_points[j]
            p_sim = combiner.optimizer.calc_similarity_sv(song_point,video_point)

            s_vals.append(song_point[1])            
            v_vals.append(video_point[1])
            s_arss.append(song_point[2])
            v_arss.append(video_point[2])
            v_imgs.append(video_point[5])
            p_sims.append(p_sim)

        d_lists = [v_imgs,s_vals,v_vals,s_arss,v_arss,p_sims]
        imb1 = plt.imread("assets/img_back1.png")

        njobs=num_cores
        ljobs = utils.get_ljobs(0,len(video_points),njobs)
        louts = Parallel(n_jobs=njobs)(
            delayed(generate_plots)(i,ftit1,imb1,d_lists,plots_dir)
                for i in ljobs)
        for lout in louts:
            for item in lout:
                video_images.append(item)
    else:
        for i in range(len(video_points)):
            utils.show_progress("Reading frames... ",i,len(video_points))
            video_images.append(cv2.imread(video_points[i][5]))

    avi_filename = redut_dir+ftit1+".avi"
    mp4_filename = redut_dir+ftit1+".mp4"

    frame = video_images[0]
    height, width, layers = frame.shape
    video = cv2.VideoWriter(avi_filename, 0, fps, (width,height))
    for i in range(len(video_images)):
        utils.show_progress("Rendering video... ",i,len(video_images))
        video.write(video_images[i])
    cv2.destroyAllWindows()
    video.release()

    inp_video = ffmpeg.input(avi_filename)
    inp_audio = ffmpeg.input(song_fnam_cut)          
    ffmpeg.concat(inp_video, inp_audio, v=1, a=1).output(mp4_filename).run()

    if(save_dir==out_dir):
        mp4_filename_f = utils.copy_file_to_out_dir(mp4_filename)
        if(global_compress_video==True):
            mp4_filename_f = utils.compress_video(mp4_filename_f)
    else:
        mp4_filename_f = utils.copy_file_to_any_dir(mp4_filename,save_dir)

    print("Done!! Out file: "+str(mp4_filename_f))

    if(os.path.exists(redut_dir_video)):
        shutil.rmtree(redut_dir_video)

def combine_pair(song_fnam,song_points,video_fnam,video_points,opt_method,result_mode):

    print("Song/video lengths: ",len(song_points),len(video_points))
    
    song_points_f, video_points_f = combiner.optimizer.run(song_points,video_points,opt_method)

    if(result_mode=="points"):
        return [song_points_f, video_points_f, len(song_points), len(video_points)]

    elif(result_mode=="score"):
        p_sims = []
        for i in range(min(len(song_points_f),len(video_points_f))):
            p_sims.append(combiner.optimizer.calc_similarity_sv(song_points_f[i],video_points_f[i]))
        m_sim = round(np.mean(p_sims),global_round_numbs)
        return m_sim
    
    elif(result_mode=="video"):
        render_video(song_fnam,song_points_f,video_fnam,video_points_f,opt_method,out_dir,False)

    elif(result_mode=="animation"):
        render_video(song_fnam,song_points_f,video_fnam,video_points_f,opt_method,out_dir,True)

    else:
        print("Error: Undefined mode "+result_mode+", exiting.")
        exit()

def calc_sims_p(i,song_list,video_fnam,video_points,sel_method):

    i1,i2,k,n = i[0],i[1],i[2],i[3]

    songsim_list = []

    for i in range(i1,i2):
        song_fnam = song_list[i]
        profgen_a = combiner.profgen.profgen("audio",64,song_fnam,False,False)
        song_points = profgen_a.get_emotion_profile()
        song_points = preprocess_points(song_points,"s")
        if(validate_pair(song_points,video_points)):
            pair_sim = combine_pair(song_fnam,song_points,video_fnam,video_points,sel_method,"score")
            print("Song "+str(i-i1)+"/"+str(i2-i1)+": "+str(song_fnam)+": "+str(pair_sim))
            songsim_list.append([song_fnam,pair_sim])

    return songsim_list

def calc_sims(video_fnam,video_points,songs_dir,shuff=True,only_cache=False):

    sel_method = "uniform"

    song_list = []
    for root, dirs, files in os.walk(songs_dir):
        for file in files:
            if file.endswith(".mp3"):
                song_list.append(os.path.join(root,file))

    if(only_cache==True):
        print("  Note: Using only songs in cache")
        song_list2 = []
        for song_fnam in song_list:
            i=-1
            while(song_fnam[i-1]!="/"):
                i-=1
            song_cdir = cache_dir+"audio/"+song_fnam[i:-4]+"/"
            if(os.path.exists(song_cdir+"labels.csv")):
                song_list2.append(song_fnam)
        song_list = song_list2

    song_amount = len(song_list)
    print("Tot Songs: "+str(song_amount))
    print("Max Songs: "+str(max_songs))

    if(shuff==True):
        random.shuffle(song_list)
    song_list = song_list[:max_songs]

    parallel = True
    if(parallel==False):
        i = [0,len(song_list),1,1]
        songsim_list = calc_sims_p(i,video_fnam,video_points,sel_method)
    else:
        njobs = num_cores
        ljobs = utils.get_ljobs(0,len(song_list),njobs)
        louts = Parallel(n_jobs=njobs)(
                delayed(calc_sims_p)(i,song_list,video_fnam,video_points,sel_method)
                    for i in ljobs)

        songsim_list = []
        for lout in louts:
            for item in lout:
                songsim_list.append(item)

    def getkey(item):
        return item[1]

    songsim_list.sort(reverse=True,key=getkey)

    return songsim_list

def make_hyperlapse(video_fnam,songs_path,opt_method,result_mode):

    profgen_v = combiner.profgen.profgen("video", 4,video_fnam,False,True)
    video_points = profgen_v.get_emotion_profile()
    video_points = preprocess_points(video_points,"v")

    if(os.path.isdir(songs_path)):
        songsim_list = calc_sims(video_fnam,video_points,songs_path)
        song_fnam = songsim_list[0][0]
    else:
        song_fnam = songs_path

    profgen_a = combiner.profgen.profgen("audio",64,song_fnam,False,True)
    song_points = profgen_a.get_emotion_profile()
    song_points = preprocess_points(song_points,"s")

    result = combine_pair(song_fnam,song_points,video_fnam,video_points,opt_method,result_mode)

    if(result_mode=="points"):
        return result
