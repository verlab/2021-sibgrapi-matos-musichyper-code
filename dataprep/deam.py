from header import *

def reject_outliers(data, m):
    return data[abs(data-np.mean(data))<m*np.std(data)]

def discretize_label_8_45(label):

    if(label<-0.45):
        label_class = 0
    elif(label<-0.30):
        label_class = 1
    elif(label<=-0.15):
        label_class = 2
    elif(label<=0.00):
        label_class = 3
    elif(label<=+0.15):
        label_class = 4
    elif(label<=+0.30):
        label_class = 5
    elif(label<=+0.45):
        label_class = 6
    else:
        label_class = 7

    return label_class

def discretize_label_generic(label,num_classes_ax,label_min,label_max):
    label_disc = None
    inc = (label_max-label_min)/num_classes_ax
    value = label_min+inc
    count = 0
    while(True):
        if(label<value):
            label_disc = count
            break
        value+=inc
        count+=1
        if(count>num_classes_ax-1):
            count=num_classes_ax-1

    return label_disc

def generate_labels(num_classes_ax,undsamp):

    data_folder = audio_dataset_dir+"DEAM/"

    #Filtering parameters
    songid_i,songid_f = 0,2060
    rejection_threshold = 0.5
    label_range_ref = 1
    valence_range = [-label_range_ref,+label_range_ref]
    arousal_range = [-label_range_ref,+label_range_ref]

    valence_folder = data_folder+"original/labels/per_rater/valence/"
    arousal_folder = data_folder+"original/labels/per_rater/arousal/"

    segment_list = []
    valence_samples = []
    arousal_samples = []

    valence_samples.append(("song-id","valence"))
    arousal_samples.append(("song-id","arousal"))

    for song_id in range(songid_i,songid_f+1):
        valence_file = valence_folder+str(song_id)+".csv"
        arousal_file = arousal_folder+str(song_id)+".csv"

        utils.show_progress("Filtering dataset... ",song_id,songid_f)

        if(os.path.exists(arousal_file) and os.path.exists(valence_file)):

            valence_data = pd.read_csv(valence_file)
            arousal_data = pd.read_csv(arousal_file)
            valence_raters_amount = valence_data.shape[0]
            arousal_raters_amount = arousal_data.shape[0]
            valence_segment_amount = valence_data.shape[1]
            arousal_segment_amount = arousal_data.shape[1]

            i_offset=1
            if(song_id>1000):
                i_offset=2

            time_index = 15000 - 500
            for i in range(valence_segment_amount-i_offset):
                time_index+=500
                str_index = "sample_"+str(time_index)+"ms"
                raters_valences = valence_data[str_index]
                raters_valences = reject_outliers(raters_valences,rejection_threshold)
                raters_mean_valence = np.mean(raters_valences)
                if not(math.isnan(raters_mean_valence)):
                    #valence_class = discretize_label_generic(raters_mean_valence,
                    #num_classes_ax,valence_range[0],valence_range[1])
                    valence_class = discretize_label_8_45(raters_mean_valence)
                    song_id_str = "songs/"+str(song_id)+".mp3"+"-"+str(time_index)
                    valence_samples.append((song_id_str,valence_class))

            time_index = 15000 - 500
            for i in range(arousal_segment_amount-i_offset):
                time_index+=500
                str_index = "sample_"+str(time_index)+"ms"
                raters_arousals = arousal_data[str_index]
                raters_arousals = reject_outliers(raters_arousals,rejection_threshold)
                raters_mean_arousal = np.mean(raters_arousals)
                if not(math.isnan(raters_mean_arousal)):
                    #arousal_class = discretize_label_generic(raters_mean_arousal,
                    #num_classes_ax,arousal_range[0],arousal_range[1])
                    arousal_class = discretize_label_8_45(raters_mean_arousal)
                    song_id_str = "songs/"+str(song_id)+".mp3"+"-"+str(time_index)
                    arousal_samples.append((song_id_str,arousal_class))

    valence_class_amount = [0]*num_classes_ax
    arousal_class_amount = [0]*num_classes_ax

    for i in range(1,len(valence_samples)):
        item = valence_samples[i]
        valence_class_amount[item[1]]+=1
    for i in range(1,len(arousal_samples)):
        item = arousal_samples[i]
        arousal_class_amount[item[1]]+=1

    if(undsamp==True):
    
        valence_samples_und = []
        arousal_samples_und = []

        valence_samples_und.append(("song-id","valence"))
        arousal_samples_und.append(("song-id","arousal"))

        min_class_samp_val = min(valence_class_amount)
        min_class_samp_ars = min(arousal_class_amount)
        
        valence_class_amount_und = [0]*num_classes_ax
        arousal_class_amount_und = [0]*num_classes_ax

        for i in range(1,len(valence_samples)):
            item = valence_samples[i]
            quad = item[1]
            if(valence_class_amount_und[quad]<min_class_samp_val):
                valence_class_amount_und[quad]+=1
                valence_samples_und.append(item)

        for i in range(1,len(arousal_samples)):
            item = arousal_samples[i]
            quad = item[1]
            if(arousal_class_amount_und[quad]<min_class_samp_ars):
                arousal_class_amount_und[quad]+=1
                arousal_samples_und.append(item)

        valence_samples = valence_samples_und
        arousal_samples = arousal_samples_und

    labels_disc_dir = data_folder+"original/labels_disc/"

    if not(os.path.exists(labels_disc_dir)):
        os.makedirs(labels_disc_dir)
    
    (pd.DataFrame(valence_samples)).to_csv(labels_disc_dir+"valence.csv",index=None,header=False)
    (pd.DataFrame(arousal_samples)).to_csv(labels_disc_dir+"arousal.csv",index=None,header=False)

def generate_quadrants(num_classes_ax):

    data_folder = audio_dataset_dir+"DEAM/"
    labels_disc_dir = data_folder+"original/labels_disc/"
    val_filename = labels_disc_dir+"valence.csv"
    ars_filename = labels_disc_dir+"arousal.csv"

    df_val = pd.read_csv(val_filename)
    df_ars = pd.read_csv(ars_filename)

    list_val = []
    list_ars = []
    list_va = []
    list_quad = []

    quad_refs = np.zeros((num_classes_ax,num_classes_ax),int)
    count=0
    for i in range(num_classes_ax):
        for j in range(num_classes_ax):
            quad_refs[i][j] = count
            count+=1
    
    for i in range(df_val.shape[0]):
        list_val.append((df_val.iloc[i,0],df_val.iloc[i,1]))
    for i in range(df_ars.shape[0]):
        list_ars.append((df_ars.iloc[i,0],df_ars.iloc[i,1]))
    
    count=0
    amount=len(list_val)
    for item_v in list_val:
        utils.show_progress("Converting into quadrants... ",count,amount)
        count+=1
        for item_a in list_ars:
            if(item_v[0]==item_a[0]):
                list_va.append((item_v[0],item_v[1],item_a[1]))

    list_quad = []
    for item in list_va:
        nam = item[0]
        val = item[1]
        ars = item[2]
        quad = quad_refs[val,ars]
        list_quad.append((nam,quad))

    (pd.DataFrame(list_quad)).to_csv(labels_disc_dir+"quadrant.csv",index=None,header=False)

def prepare_labels(num_classes_ax,clean,undsamp):

    data_folder = audio_dataset_dir+"DEAM/"
    labels_disc_dir = data_folder+"original/labels_disc/"

    if(clean==True):
        generate_labels(num_classes_ax,undsamp)
        generate_quadrants(num_classes_ax)

    for label_type in ["valence","arousal","quadrant"]:
        lab_filename = labels_disc_dir+label_type+".csv"    
        shutil.copyfile(lab_filename,data_folder+"labels_"+label_type[0]+".csv")

def run(num_classes=64,clean=False,undsamp=True):
    num_classes_ax = int(math.sqrt(num_classes))
    prepare_labels(num_classes_ax,clean,undsamp)