from header import *

def run(video_fnam,bas_method):

    if not(bas_method in ["msh","sas","ours"]):
        print("Unlisted method, exiting.")
        exit()

    basecomp_dir = "/srv/storage/datasets/Diognei/Cache/basecomp/v7/"
    basecomp_files_dir = "/srv/storage/datasets/Diognei/Video/Baselines/"

    songs_csv = basecomp_dir+"songs_fix.csv"

    songs_list = []

    df = pd.read_csv(songs_csv,header=None)
    for i in range(df.shape[0]):
        songs_list.append([df.iloc[i][0],df.iloc[i][1]])

    basmethod_dir = basecomp_dir+bas_method+"/"
    if not(os.path.exists(basmethod_dir)):
        os.mkdir(basmethod_dir)
    
    points_dir = basmethod_dir+"points/"
    if not(os.path.exists(points_dir)):
        os.mkdir(points_dir)

    video_points_dir = points_dir+utils.path2fnam(video_fnam)[:-4]+"/"
    if not(os.path.exists(video_points_dir)):
        os.mkdir(video_points_dir)

    profgen_v = combiner.profgen.profgen("video", 4,video_fnam,False,True)
    orig_vidpts = profgen_v.get_emotion_profile()

    emosim_list = []
    spdrat_list = []
    vidfid_list = []
    vidshk_list = []

    for i in range(len(songs_list)):
        fsong_fnam = songs_list[i][0]
        fsong_len = songs_list[i][1]

        ivideo_len = len(orig_vidpts)

        des_spd = ivideo_len/fsong_len
        des_spd_int = int(des_spd)+1
        
        msh_out_dir = basecomp_files_dir+"MSH_Output/"
        fvideo_fnam = msh_out_dir+utils.path2fnam(video_fnam)[:-4]+"_hyperlapse_"+str(des_spd_int)+"x_std.mp4"

        vidpts_csv = video_points_dir+utils.path2fnam(fsong_fnam)[:-4]+"_v.csv"
        sngpts_csv = video_points_dir+utils.path2fnam(fsong_fnam)[:-4]+"_s.csv"

        if (not(os.path.exists(vidpts_csv)) or not(os.path.exists(sngpts_csv))):

            if(bas_method=="msh"):

                msh_out_dir = basecomp_files_dir+"MSH_Output/"
                
                fvideo_fnam = msh_out_dir+utils.path2fnam(video_fnam)[:-4]+"_hyperlapse_"+str(des_spd_int)+"x_std.mp4"
                if not(os.path.exists(fvideo_fnam)):
                    fvideo_fnam = msh_out_dir+utils.path2fnam(video_fnam)[:-4]+"_hyperlapse_"+str(des_spd_int-1)+"x_std.mp4"

                profgen_v = combiner.profgen.profgen("video", 4,fvideo_fnam,False,True)
                vidpts = profgen_v.get_emotion_profile()

                profgen_a = combiner.profgen.profgen("audio",64,fsong_fnam,False,False)
                sngpts = profgen_a.get_emotion_profile()
                sngpts = sngpts[:int(len(orig_vidpts)/des_spd_int)]

            elif(bas_method=="sas"):

                sas_out_dir = basecomp_files_dir+"out/"

                selfms_fnam = sas_out_dir+utils.path2fnam(video_fnam)[:-4]+"_LLC_EXP_"+str(i+1)+".csv"

                sel_frames = []
                df = pd.read_csv(selfms_fnam)
                for i in range(df.shape[0]):
                    sel_frames.append(int(df.iloc[i][0]))

                vidpts = []
                for i in range(len(sel_frames)):
                    vidpts.append(orig_vidpts[sel_frames[i]])

                profgen_a = combiner.profgen.profgen("audio",64,fsong_fnam,False,False)
                sngpts = profgen_a.get_emotion_profile()
                sngpts = sngpts[:int(len(orig_vidpts)/des_spd_int)]

            elif(bas_method=="ours"):

                profgen_a = combiner.profgen.profgen("audio",64,fsong_fnam,False,False)
                sngpts = profgen_a.get_emotion_profile()
                #sngpts = sngpts[:int(len(orig_vidpts)/des_spd_int)]

                print("Parameters: ",ivideo_len,des_spd,des_spd_int,len(sngpts),len(orig_vidpts)/len(sngpts))

                sngpts, vidpts = combiner.optimizer.run(sngpts, orig_vidpts, "ours")

            (pd.DataFrame(vidpts)).to_csv(vidpts_csv, index=None, header=False)
            (pd.DataFrame(sngpts)).to_csv(sngpts_csv, index=None, header=False)

        else:
            print("  Note: using cache generated points")

        vidpts = []
        df = pd.read_csv(vidpts_csv,header=None)
        for i in range(df.shape[0]):
            vidpt = []
            for j in range(df.shape[1]):
                vidpt.append(df.iloc[i][j])
            vidpts.append(vidpt)

        sngpts = []
        df = pd.read_csv(sngpts_csv,header=None)
        for i in range(df.shape[0]):
            sngpt = []
            for j in range(df.shape[1]):
                sngpt.append(df.iloc[i][j])
            sngpts.append(sngpt)

        fvideo_len = len(vidpts)
        #desspd = des_spd_int
        desspd = len(orig_vidpts)/len(sngpts)
        obtspd = ivideo_len/fvideo_len

        #if(len(vidpts)>len(sngpts)):
        #    vidpts = vidpts[0:int(len(sngpts))]
        #elif(len(sngpts)>len(vidpts)):
        #    sngpts = sngpts[0:int(len(vidpts))]

        orig_vidpts = orig_vidpts[4:5]
        vidpts = vidpts[4:5]

        print(len(vidpts))
        print(len(orig_vidpts))

        #emosim = evaluator.metrics.calc_emosim(sngpts,vidpts)
        #spdrat = max(desspd,obtspd)/min(desspd,obtspd)
        #fidscr = evaluator.metrics.calc_vidfid(vidpts,video_fnam)
        #vidshk = evaluator.metrics.calc_vidshk(vidpts)

        fidscr = evaluator.metrics.calc_vidfid2(orig_vidpts,vidpts)

        print(video_fnam)
        print(fvideo_fnam)
        print(fidscr)
        exit()

        print("Scores: ",emosim,spdrat,fidscr,vidshk)

        emosim_list.append(emosim)
        spdrat_list.append(spdrat)
        vidfid_list.append(fidscr)
        vidshk_list.append(vidshk)

    emosim_list.append(round(np.mean(emosim_list),global_round_numbs))
    spdrat_list.append(round(np.mean(spdrat_list),global_round_numbs))
    vidfid_list.append(round(np.mean(vidfid_list),global_round_numbs))
    vidshk_list.append(round(np.mean(vidshk_list),global_round_numbs))

    results_list = []
    for i in range(len(emosim_list)):
        results_list.append([emosim_list[i],spdrat_list[i],vidfid_list[i],vidshk_list[i]])

    print(results_list)

    basemeth_dir = basecomp_dir+bas_method+"/"

    results_csv = basemeth_dir+utils.path2fnam(video_fnam)[:-4]+".csv"
    (pd.DataFrame(results_list)).to_csv(results_csv, index=None, header=False)