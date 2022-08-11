from header import *
import scipy.stats

score_names_ref = [
    "emotion_similarity",
    "song_truncation",
    "video_truncation",
    "desired_speedup",
    "obtained_speedup",
    "video_smoothness",
    "video_shaking",
    "video_fidscore"
    ]

def calc_mean_ci(data,confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se*scipy.stats.t.ppf((1+confidence)/2.,n-1)
    m_f = round(m,global_round_numbs)
    sd_f =  round(np.std(a),global_round_numbs)
    ci_f = str(round(m,global_round_numbs))+" ± "+str(round(h,global_round_numbs))
    return m_f, sd_f, ci_f

def read_points_from_csv(points_csv):
    df = pd.read_csv(points_csv, header=None)
    song_points = []
    video_points = []
    song_fnam = df.iloc[0][0]
    video_fnam = df.iloc[0][1]
    len_sp_i = df.iloc[0][2]
    len_vp_i = df.iloc[0][3]
    for i in range(1,df.shape[0]):
        utils.show_progress("Reading points... ",i,df.shape[0])
        song_point = [0]*5
        video_point = [0]*6
        song_point[0] = int(df.iloc[i][0])
        song_point[1] = float(df.iloc[i][1])
        song_point[2] = float(df.iloc[i][2])
        song_point[3] = float(df.iloc[i][3])
        song_point[4] = float(df.iloc[i][4])     

        video_point[0] = int(df.iloc[i][5])
        video_point[1] = float(df.iloc[i][6])
        video_point[2] = float(df.iloc[i][7])
        video_point[3] = float(df.iloc[i][8])
        video_point[4] = float(df.iloc[i][9])
        video_point[5] = str(df.iloc[i][10])

        song_points.append(song_point)
        video_points.append(video_point)
        
    return song_fnam, video_fnam, len_sp_i, len_vp_i, song_points, video_points

def generate_video(method_dir,opt_method):

    song_fnam, video_fnam, song_points, video_points = read_points_from_csv(method_dir+"points.csv")
    combiner.hypmaker.render_video(song_fnam,song_points,video_fnam,video_points,opt_method,method_dir,False)

def calc_method_scores(points_csv,scores_csv):

    song_fnam, video_fnam, len_sp_i, len_vp_i, song_points, video_points = read_points_from_csv(points_csv)

    emosim = evaluator.metrics.calc_emosim(song_points,video_points)
    sngtrc = evaluator.metrics.calc_sngtrc(song_points,len_sp_i)
    vidtrc = evaluator.metrics.calc_vidtrc(video_points,len_vp_i)
    desspd = evaluator.metrics.calc_desspd(len_sp_i,len_vp_i)
    obtspd = evaluator.metrics.calc_obtspd(video_points,len_vp_i)
    vidsmt = evaluator.metrics.calc_vidsmt(video_points)
    vidshk = evaluator.metrics.calc_vidshk(video_points)
    fidscr = evaluator.metrics.calc_vidfid(video_points,video_fnam)

    score_names = score_names_ref
    score_values = [emosim,sngtrc,vidtrc,desspd,obtspd,vidsmt,vidshk,fidscr]
    score_table = [score_names,score_values]

    (pd.DataFrame(score_table)).to_csv(scores_csv, index=None, header=False)

def calc_video_scores(video_dir,get_table=False):

    print("Generating video scores table...")

    opt_class_dict = combiner.optimizer.get_opt_class_dict()

    score_names = ["metric"]+score_names_ref
    score_table = []
    score_table.append(score_names)
    for opt_method in opt_class_dict:
        method_scores = [opt_method]
        for k in range(len(score_names_ref)):
            method_scores.append([])
        score_table.append(method_scores)

    if(global_video_songs_mode=="bests"):
        video_songs_csv = video_dir+"songs_sel.csv"
    elif(global_video_songs_mode=="fixed"):
        video_songs_csv = video_dir+"songs_fix.csv"
    df_sn = pd.read_csv(video_songs_csv, header=None)
    song_exists = False
    for i in range(df_sn.shape[0]):
        song_dir = video_dir+utils.path2fnam(df_sn.iloc[i][0])[:-4]+"/"
        if(os.path.exists(song_dir)):
            for j in range(1,len(score_table)):
                opt_method = score_table[j][0]
                method_dir = song_dir+opt_method+"/"
                ms_csv = method_dir+"scores.csv"
                if(os.path.exists(ms_csv)):
                    song_exists = True
                    df_ms = pd.read_csv(ms_csv)
                    for k in range(1,len(score_names)):
                        score_table[j][k].append(float(df_ms.iloc[0][k-1]))

    if(get_table==True):
        if(song_exists==True):
            return score_table
        else:
            return None

    score_table2 = []
    score_table2.append(score_table[0])
    score_table2.append(["method"]+(["mean","sd"]*len(score_names_ref)))
    for i in range(1,len(score_table)):
        sc_line = [score_table[i][0]]
        for j in range(1,len(score_table[i])):
            mean,sd,ci = calc_mean_ci(score_table[i][j])
            sc_line+=[mean,sd]
        score_table2.append(sc_line)
    score_table = score_table2

    scores_csv = video_dir+"scores.csv"
    (pd.DataFrame(score_table)).to_csv(scores_csv, index=None, header=False)

    for item in score_table:
        print(item)

def process_method(song_dir,video_fnam,song_fnam,opt_method):

    print("Running experiments for method "+opt_method)
    
    method_dir = song_dir+opt_method+"/"
    
    if not(os.path.exists(method_dir)):
        os.mkdir(method_dir)

    points_csv = method_dir+"points.csv"
    if not os.path.exists(points_csv):
        result = combiner.hypmaker.make_hyperlapse(video_fnam,song_fnam,opt_method,"points")
        song_points, video_points, len_sp_i, len_vp_i = result[0], result[1], result[2], result[3]

        if(len(song_points)!=len(video_points)):
            print("Error: Sizes mismatch, exiting.")
            exit()

        all_points_f = []
        all_points_f.append([song_fnam,video_fnam,len_sp_i,len_vp_i,-1,-1,-1,-1,-1,-1,-1])
        for i in range(len(song_points)):
            song_point = song_points[i]
            video_point = video_points[i]
            song_video_point = []
            for j in range(len(song_point)):
                song_video_point.append(song_point[j])
            for j in range(len(video_point)):
                song_video_point.append(video_point[j])
            all_points_f.append(song_video_point)

        (pd.DataFrame(all_points_f)).to_csv(points_csv, index=None, header=False)
    else:
        print("  Note: Using cache points")

    scores_csv = method_dir+"scores.csv"
    if not(os.path.exists(scores_csv)):
        calc_method_scores(points_csv,scores_csv)

    if(global_genvid_in_runner==True):
        generate_video(method_dir,opt_method)

def process_song(video_dir,video_fnam,song_fnam):

    print("Running experiments for song "+song_fnam)

    song_dir = video_dir+utils.path2fnam(song_fnam)[:-4]+"/"

    if not(os.path.exists(song_dir)):
        os.mkdir(song_dir)

    opt_class_dict = combiner.optimizer.get_opt_class_dict()

    for opt_method in opt_class_dict:
        process_method(song_dir,video_fnam,song_fnam,opt_method)

def process_video(evaluator_dir,video_fnam,songs_dir,clean_level=0):

    print("Running experiments for video "+video_fnam)

    video_dir = evaluator_dir+utils.path2fnam(video_fnam)[:-4]+"/"

    if(clean_level==3):
        opt = input("Do you really want to delete all the previous tables (y/n)? ")
        if(opt=="y"):
            if(os.path.exists(video_dir)):
                print("Note: Cleaning previous results")
                shutil.rmtree(video_dir)
    
    if(clean_level==2):
        file_list = []
        for root, dirs, files in os.walk(video_dir):
            for file in files:
                if(file=="points.csv" or file=="scores.csv" or file.endswith(".mp4")):
                    file_list.append(os.path.join(root, file))
        for fnam in file_list:
            if(os.path.exists(fnam)):
                os.remove(fnam)

    if(clean_level==1):
        file_list = []
        for root, dirs, files in os.walk(video_dir):
            for file in files:
                if(file=="scores.csv"):
                    file_list.append(os.path.join(root, file))
        for fnam in file_list:
            if(os.path.exists(fnam)):
                os.remove(fnam)

    if not(os.path.exists(video_dir)):
        os.mkdir(video_dir)

    print("  Selecting songs...")

    if(global_video_songs_mode=="bests"):
        video_songs_csv = video_dir+"songs_sel.csv"    
        if not(os.path.exists(video_songs_csv)):
            profgen_v = combiner.profgen.profgen("video", 4,video_fnam,False,True)
            video_points = profgen_v.get_emotion_profile()
            video_points = combiner.hypmaker.preprocess_points(video_points,"v")
            songsim_list = combiner.hypmaker.calc_sims(video_fnam,video_points,songs_dir)
            (pd.DataFrame(songsim_list)).to_csv(video_songs_csv, index=None, header=False)
        else:
            print("  Note: Using cache selected songs")
    elif(global_video_songs_mode=="fixed"):
        video_songs_csv = video_dir+"songs_fix.csv"
        if not(os.path.exists(video_songs_csv)):
            fixed_songs_dir = audio_dataset_dir+"MSHP/Fixed/"
            ltmp1 = []
            for root, dirs, files in os.walk(fixed_songs_dir):
                for file in files:
                    if file.endswith(".mp3"):
                        ltmp1.append(os.path.join(root, file))
            songlen_list = []
            for sntmp1 in ltmp1:
                profgen_a = combiner.profgen.profgen("audio",64,sntmp1,False,False)
                sngpts = profgen_a.get_emotion_profile()
                songlen_list.append([sntmp1,len(sngpts)])                
            (pd.DataFrame(songlen_list)).to_csv(video_songs_csv, index=None, header=False)
        else:
            print("  Note: Using cache selected songs")
            
    df = pd.read_csv(video_songs_csv, header=None)

    sel_songs = []
    for i in range(df.shape[0]):
        sel_songs.append(df.iloc[i][0])
    sel_songs = sel_songs[:global_songs_per_video]

    for song_fnam in sel_songs:
        process_song(video_dir,video_fnam,song_fnam)
    
    calc_video_scores(video_dir)

def run_experiments(video_fnam,clean):

    video_list = []

    if(os.path.isdir(video_fnam)):
        for root, dirs, files in os.walk(video_fnam):
            for file in files:
                if file.endswith(".mp4"):
                    video_list.append(os.path.join(root, file))
    else:
        video_list.append(video_fnam)

    for video_fnam in video_list:

        songs_dir = audio_dataset_dir + "MSHP/"

        evaluator_dir = cache_dir+"evaluator/"
        if not(os.path.exists(evaluator_dir)):
            os.mkdir(evaluator_dir)
        
        process_video(evaluator_dir,video_fnam,songs_dir,clean)

def create_full_table():

    evaluator_dir = cache_dir+"evaluator/"
    results_csv = evaluator_dir+"results.csv"

    if not(os.path.exists(evaluator_dir)):
        print("Results directory not exists, exiting")
        exit()
    
    video_dirs = [
        os.path.join(evaluator_dir, subdir+"/") 
        for subdir in os.listdir(evaluator_dir) 
        if os.path.isdir(os.path.join(evaluator_dir,subdir))
        ]

    full_table = []
    score_names_ref_p = []
    for item in score_names_ref:
        score_names_ref_p+=[item,""]
    full_table.append(["video","method"]+score_names_ref_p)
    full_table.append(["video","method"]+(["mean","sd"]*len(score_names_ref)))
    for video_dir in video_dirs:
        i=-2
        while(video_dir[i]!="/"):
            i-=1
        video_title = video_dir[i+1:-1]
        score_table = calc_video_scores(video_dir,True)
        if not(score_table is None):
            for i in range(1,len(score_table)):
                st_line = score_table[i]
                ft_line = [video_title]+st_line
                full_table.append(ft_line)

    opt_class_dict = combiner.optimizer.get_opt_class_dict()

    mean_scores = []
    for opt_method in opt_class_dict:
        mean_lists = []
        for i in range(len(score_names_ref)):
            mean_lists.append([])
        mean_scores.append(["Total",opt_method]+mean_lists)
        
    for i in range(2,len(full_table),len(opt_class_dict)):
        for j in range(i,i+len(opt_class_dict)):
            ft_line = full_table[j][0:2]
            for k in range(2,len(score_names_ref)+2):
                mean_scores[j-i][k] = mean_scores[j-i][k] + full_table[j][k]
                mean,sd,ci = calc_mean_ci(full_table[j][k])
                ft_line+=[mean,sd]
            full_table[j] = ft_line

    for i in range(len(mean_scores)):
        ft_line = mean_scores[i][0:2]
        for j in range(2,len(score_names_ref)+2):
            mean,sd,ci = calc_mean_ci(mean_scores[i][j])
            ft_line+=[mean,sd]
        full_table.append(ft_line)
        
    (pd.DataFrame(full_table)).to_csv(results_csv, index=None, header=False)

    for item in full_table:
        print(item)