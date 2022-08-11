from header import *

def draw_point(ax,x,y,color,marker):
    x = (x*(+0.78))*400+400
    y = (y*(-0.78))*400+400
    ax.plot(x,y,color=color,marker=marker)

def generate_video(video_name,song_name):

    print("Files: ",video_name,song_name)

    basecomp_dir = cache_dir+"basecomp/v4/"
    imb1 = plt.imread("assets/img_back1.png")

    read_fact=1.0
    quick_test=True
    if(quick_test==True):
        read_fact=1.0

    method_inds = ["sas","ours"]
    method_names = ["SASv2","Ours"]

    sngpts_list = [None]*len(method_inds)
    vidpts_list = [None]*len(method_inds)

    for k in range(len(method_inds)):

        method = method_inds[k]

        vps_csv = basecomp_dir+method+"/points/"+video_name+"/"+song_name+"_v.csv"
        vps = []
        df = pd.read_csv(vps_csv,header=None)
        for i in range(int(df.shape[0]*read_fact)):
            utils.show_progress("["+str(method)+"] "+"Reading video points... ",i,df.shape[0])
            vp = []
            for j in range(df.shape[1]):
                vp.append(df.iloc[i][j])
            vps.append(vp)

        sps_csv = basecomp_dir+method+"/points/"+video_name+"/"+song_name+"_s.csv"
        sps = []
        df = pd.read_csv(sps_csv,header=None)
        for i in range(int(df.shape[0]*read_fact)):
            sp = []
            utils.show_progress("["+str(method)+"] "+"Reading song points... ",i,df.shape[0])
            for j in range(df.shape[1]):
                sp.append(df.iloc[i][j])
            sps.append(sp)

        if(len(vps)>len(sps)):
            vps = vps[0:int(len(sps))]
        elif(len(sps)>len(vps)):
            sps = sps[0:int(len(vps))]

        sngpts_list[k] = sps
        vidpts_list[k] = vps

    def generate_image(j,ftit1,imb1):

        j1,j2,jk,jn = j[0],j[1],j[2],j[3]

        out_imgs = []

        print("  Init part "+str(jk)+"/"+str(jn))

        for j in range(j1,j2):

            metsep = 5

            fig = plt.figure(figsize=((10*(metsep+128))/72, 10))
            fig.suptitle(ftit1, fontsize=16)

            grid = plt.GridSpec(72,metsep+128)

            grid.update(left=None, bottom=None, right=None, wspace=5.0, hspace=1.0)

            method_inds = ["sas","ours"]
            method_names = ["SASv2","Ours"]

            ax_frms = [None,None]
            ax_emos = [None,None]
            ax_vals = [None,None]
            ax_arss = [None,None]
            ax_sims = [None,None]

            ax_frms[0] = fig.add_subplot(grid[  0: 48,  0: 64])    
            ax_vals[0] = fig.add_subplot(grid[ 48: 56,  0: 40])
            ax_arss[0] = fig.add_subplot(grid[ 56: 64,  0: 40])
            ax_sims[0] = fig.add_subplot(grid[ 64: 72,  0: 40])
            ax_emos[0] = fig.add_subplot(grid[ 48: 72, 40: 64])

            ax_frms[1] = fig.add_subplot(grid[  0: 48,metsep+ 64:metsep+128])    
            ax_vals[1] = fig.add_subplot(grid[ 48: 56,metsep+ 64:metsep+104])
            ax_arss[1] = fig.add_subplot(grid[ 56: 64,metsep+ 64:metsep+104])
            ax_sims[1] = fig.add_subplot(grid[ 64: 72,metsep+ 64:metsep+104])
            ax_emos[1] = fig.add_subplot(grid[ 48: 72,metsep+104:metsep+128])

            ax_list = []
            ax_list+=[ax_frms[0],ax_vals[0],ax_arss[0],ax_sims[0],ax_emos[0]]
            ax_list+=[ax_frms[1],ax_vals[1],ax_arss[1],ax_sims[1],ax_emos[1]]

            for k in range(len(method_inds)):

                vidpts = vidpts_list[k]
                sngpts = sngpts_list[k]

                s_vals = []
                s_arss = []
                v_vals = []
                v_arss = []
                p_sims = []

                scltmp1=1.8

                for i in range(len(sngpts)):
                    s_vals.append(scltmp1*sngpts[i][1]*0.5)
                    s_arss.append(scltmp1*sngpts[i][2]*0.5)

                for i in range(len(vidpts)):
                    v_vals.append(scltmp1*vidpts[i][1])
                    v_arss.append(scltmp1*vidpts[i][2])

                dif_len = abs(len(vidpts)-len(sngpts))
                min_len = min(len(vidpts),len(sngpts))
                max_len = max(len(vidpts),len(sngpts))

                ax_vals[k].set_xlim(0,max_len)
                ax_arss[k].set_xlim(0,max_len)
                ax_sims[k].set_xlim(0,max_len)

                limtmp1 = 1.2

                for i in range(min(len(vidpts),len(sngpts))):
                    p_sims.append(combiner.optimizer.calc_similarity_sv(sngpts[i],vidpts[i],mode="disc")) 
                m_sim = round(np.mean(p_sims[:j]),global_round_numbs)

                ax_frms[k].set_title(str(method_names[k]))
                ax_frms[k].set_xticks([])
                ax_frms[k].set_yticks([])
                ax_frms[k].imshow(plt.imread(vidpts[j][5]))
                
                ax_emos[k].set_xticks([])
                ax_emos[k].set_yticks([])
                ax_emos[k].set_xlabel("Valence-Asoural Plane")
                ax_emos[k].imshow(imb1)

                #draw_point(ax_emos[k],s_vals[j],s_arss[j],"green","o")
                #draw_point(ax_emos[k],v_vals[j],v_arss[j],"blue","o")
                for i in range(0,j):
                    if(i<len(p_sims)):
                        draw_point(ax_emos[k],s_vals[i],s_arss[i],(1.0,0,0,p_sims[i]**20),"o")

                if(True):
                    ax_vals[k].set_ylabel("Valence")
                    ax_arss[k].set_ylabel("Arousal")
                    ax_sims[k].set_ylabel("Similarity")
                else:
                    ax_vals[k].set_yticks([])
                    ax_arss[k].set_yticks([])
                    ax_sims[k].set_yticks([])

                ax_vals[k].set_xticks([])
                ax_vals[k].set_ylim([-limtmp1, +limtmp1])
                ax_vals[k].plot(s_vals,label="audio",color="green")
                ax_vals[k].plot(v_vals,label="video",color="blue")
                ax_vals[k].axvline(j, c='m')
        
                ax_arss[k].set_xticks([])
                ax_arss[k].set_ylim([-limtmp1, +limtmp1])
                ax_arss[k].plot(s_arss,label="audio",color="green")
                ax_arss[k].plot(v_arss,label="video",color="blue")
                ax_arss[k].axvline(j, c='m')
                
                ax_sims[k].set_xlabel("Time")
                ax_sims[k].set_xticks([])
                ax_sims[k].set_ylim([-0.2, +limtmp1])
                ax_sims[k].plot(p_sims,label="average: "+str(m_sim),color="red")
                ax_sims[k].legend(loc='lower left')
                ax_sims[k].axvline(j, c='m')

            #fntmp1="out/qres_"+ftit1+".png"
            #plt.savefig(fntmp1,bbox_inches='tight')
            #print("Done! Out file: "+str(fntmp1))

            cv2img = utils.plt2array(fig)
            height, width, channels = cv2img.shape
            xtmp1 = int(width*0.075)
            xtmp2 = int(width-width*0.075)
            cv2img = cv2img[:,xtmp1:xtmp2]
            height, width, channels = cv2img.shape
            cv2img = cv2.resize(cv2img,(int(width*0.75),int(height*0.75)))

            out_imgs.append(cv2img)

            for ax in ax_list:
                ax.clear()
            plt.close()
        
        print("  Done part "+str(jk)+"/"+str(jn))

        return out_imgs

    ftit1 = "Video: "+video_name+"\n Song: "+song_name
    
    min_len = min(len(vidpts_list[0]),len(sngpts_list[0]),len(vidpts_list[1]),len(sngpts_list[1]))

    for k in range(len(method_inds)):
        sngpts_list[k] = sngpts_list[k][:min_len]
        vidpts_list[k] = vidpts_list[k][:min_len]

    video_images = []

    print("Generating plots...")
    
    njobs = num_cores
    ljobs = utils.get_ljobs(0,min_len,njobs)
    louts = Parallel(n_jobs=njobs)(
        delayed(generate_image)(j,ftit1,imb1)
            for j in ljobs)
    for lout in louts:
        for item in lout:
            video_images.append(item)
    
    avi_filename = out_dir+video_name+"_"+song_name+".avi"
    mp4_filename = avi_filename[:-4]+".mp4"

    import ffmpeg
    import librosa

    fps = global_out_fps
    
    frame = video_images[0]
    height, width, layers = frame.shape
    video = cv2.VideoWriter(avi_filename, 0, fps, (width,height))
    for i in range(len(video_images)):
        utils.show_progress("Rendering video... ",i,len(video_images))
        video.write(video_images[i])
    video.release()
    
    song_fnam = audio_dataset_dir+"MSHP/Fixed/"+song_name+".mp3"

    song_fnam_cut = out_dir+"song_piece.wav"
    scut_len = min_len/fps
    y, sr = librosa.load(song_fnam,offset=0,duration=scut_len,mono=False)
    librosa.output.write_wav(song_fnam_cut,y,sr)

    inp_video = ffmpeg.input(avi_filename)
    inp_audio = ffmpeg.input(song_fnam_cut)          
    ffmpeg.concat(inp_video, inp_audio, v=1, a=1).output(mp4_filename).run()

    mp4_filename_c = utils.compress_video(mp4_filename)

    print("Done!! Out file: "+str(mp4_filename_c))
    
def run():

    pair_id = 3

    if(pair_id==1):
        video_name, song_name = "Bike3", "InTheEnd"
    elif(pair_id==2):
        video_name, song_name = "MontOldCity1", "OnwardToFreedom"
    elif(pair_id==3):
        video_name, song_name = "Walking4", "LittleTalks"

    generate_video(video_name,song_name)