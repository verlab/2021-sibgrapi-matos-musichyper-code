from header import *

def draw_point(ax,x,y,color,marker):
    x = (x*(+0.78))*400+400
    y = (y*(-0.78))*400+400
    ax.plot(x,y,color=color,marker=marker)

def plot_pair(fig_index,video_name,song_name,all_met):

    print("Generating plots...")

    ftit1 = video_name+"-"+song_name
    ftit2 = "fig_qrescomp1x"+str(fig_index)

    fig = plt.figure(figsize=(20, 10))
    #fig.suptitle(ftit1, fontsize=16)
    grid = plt.GridSpec(8,3)
    grid.update(left=None, bottom=None, right=None, wspace=0.2, hspace=0.1)

    method_list = ["msh","sas","ours"]
    method_list_b = ["MSH","SASv2","Ours"]
    if(all_met==False):
        method_list = ["ours"]
        method_list_b = ["Ours"]

    ax_emos = [None,None,None]
    ax_vals = [None,None,None]
    ax_arss = [None,None,None]
    ax_sims = [None,None,None]

    for k in range(3):
        ax_emos[k] = fig.add_subplot(grid[0:5,k:k+1])
        ax_vals[k] = fig.add_subplot(grid[5:6,k:k+1])
        ax_arss[k] = fig.add_subplot(grid[6:7,k:k+1])
        ax_sims[k] = fig.add_subplot(grid[7:8,k:k+1])

    imb1 = plt.imread("assets/img_back1.png")

    basecomp_dir = cache_dir+"basecomp/v3/"

    vidpts_list = []
    sngpts_list = []

    fplen = float("inf")

    for k in range(len(method_list)):

        read_fact=1.0

        method = method_list[k]

        vidpts_csv = basecomp_dir+method+"/points/"+video_name+"/"+song_name+"_v.csv"
        vidpts = []
        df = pd.read_csv(vidpts_csv,header=None)
        for i in range(int(df.shape[0]*read_fact)):
            utils.show_progress("["+str(method)+"] "+"Reading video points... ",i,df.shape[0])
            vidpt = []
            for j in range(df.shape[1]):
                vidpt.append(df.iloc[i][j])
            vidpts.append(vidpt)

        sngpts_csv = basecomp_dir+method+"/points/"+video_name+"/"+song_name+"_s.csv"
        sngpts = []
        df = pd.read_csv(sngpts_csv,header=None)
        for i in range(int(df.shape[0]*read_fact)):
            sngpt = []
            utils.show_progress("["+str(method)+"] "+"Reading song points... ",i,df.shape[0])
            for j in range(df.shape[1]):
                sngpt.append(df.iloc[i][j])
            sngpts.append(sngpt)

        if(len(vidpts)>len(sngpts)):
            vidpts = vidpts[0:int(len(sngpts))]
        elif(len(sngpts)>len(vidpts)):
            sngpts = sngpts[0:int(len(vidpts))]
        
        vidpts_list.append(vidpts)
        sngpts_list.append(sngpts)

        fplen = min(fplen,len(vidpts),len(sngpts))

    for k in range(len(method_list)):

        vidpts = vidpts_list[k][:fplen]
        sngpts = sngpts_list[k][:fplen]

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
            v_vals.append(scltmp1*vidpts[i][3])
            v_arss.append(scltmp1*vidpts[i][4])

        dif_len = abs(len(vidpts)-len(sngpts))
        min_len = min(len(vidpts),len(sngpts))
        max_len = max(len(vidpts),len(sngpts))

        ax_vals[k].set_xlim(0,max_len)
        ax_arss[k].set_xlim(0,max_len)
        ax_sims[k].set_xlim(0,max_len)

        limtmp1 = 1.2

        for i in range(min(len(vidpts),len(sngpts))):
            p_sims.append(combiner.optimizer.calc_similarity_sv(sngpts[i],vidpts[i]))
        m_sim = round(np.mean(p_sims),global_round_numbs)
        d_sim = round( np.std(p_sims),global_round_numbs)
        
        ax_emos[k].set_title(str(method_list_b[k]))
        ax_emos[k].set_xticks([])
        ax_emos[k].set_yticks([])
        ax_emos[k].imshow(imb1)
        ax_emos[k].set_ylabel("Emotions (red intensity = similarity)")

        for i in range(len(sngpts)):
            #utils.show_progress("["+str(method)+"] "+"Plotting song emotions... ",i,len(sngpts))
            #draw_point(ax_emos[k],s_vals[i],s_arss[i],(0,0.5,0,0.05),"o")
            if(i<len(p_sims)):
                draw_point(ax_emos[k],s_vals[i],s_arss[i],(1.0,0,0,p_sims[i]**20),"o")
        #for i in range(len(vidpts)):
            #utils.show_progress("["+str(method)+"] "+"Plotting video emotions... ",i,len(vidpts))
            #draw_point(ax_emos[k],v_vals[i],v_arss[i],(0,0,1.0,0.05),"o")
            #if(i<len(p_sims)):
            #    draw_point(ax_emos[k],v_vals[i],v_arss[i],(1.0,0,0,p_sims[i]**20),"o")

        ax_vals[k].set_ylabel("Valence (X)")
        ax_vals[k].set_xticks([])
        ax_vals[k].set_ylim([-limtmp1, +limtmp1])
        ax_vals[k].plot(s_vals,label="audio",color="green")
        ax_vals[k].plot(v_vals,label="video",color="blue")
   
        ax_arss[k].set_ylabel("Arousal (Y)")
        ax_arss[k].set_xticks([])
        ax_arss[k].set_ylim([-limtmp1, +limtmp1])
        ax_arss[k].plot(s_arss,label="audio",color="green")
        ax_arss[k].plot(v_arss,label="video",color="blue")
        
        ax_sims[k].set_ylabel("Similarity")
        ax_sims[k].set_xlabel("Frame id")
        ax_sims[k].set_ylim([-0.2, +limtmp1])
        ax_sims[k].plot(p_sims,label="mean: "+str(m_sim),color="red")
        ax_sims[k].legend(loc='lower left') 

    print("Creating figure... ")

    fntmp1="out/"+ftit2+".png"
    plt.savefig(fntmp1,bbox_inches='tight',dpi=300)
    print("Done! Out file: "+str(fntmp1),)
    #plt.show()

def plot_bssres(index,video_name):

    print("Generating plots...")

    fig = plt.figure(figsize=(20, 10))
    #fig.suptitle(ftit1, fontsize=16)
    grid = plt.GridSpec(3,2)
    grid.update(left=None, bottom=None, right=None, wspace=0.1, hspace=0.1)

    ax_emos = fig.add_subplot(grid[0:3,0])
    ax_vals = fig.add_subplot(grid[0:1,1])
    ax_arss = fig.add_subplot(grid[1:2,1])
    ax_sims = fig.add_subplot(grid[2:3,1])

    read_fact=1.0

    method = "ours"

    imb1 = plt.imread("assets/img_back1.png")

    bssres_dir = cache_dir+"bssres/5000/"

    vidpts_csv = bssres_dir+video_name+"/points_v.csv"
    vidpts = []
    df = pd.read_csv(vidpts_csv,header=None)
    for i in range(int(df.shape[0]*read_fact)):
        utils.show_progress("["+str(method)+"] "+"Reading video points... ",i,df.shape[0])
        vidpt = []
        for j in range(df.shape[1]):
            vidpt.append(df.iloc[i][j])
        vidpts.append(vidpt)

    sngpts_csv = bssres_dir+video_name+"/points_s.csv"
    sngpts = []
    df = pd.read_csv(sngpts_csv,header=None)
    for i in range(int(df.shape[0]*read_fact)):
        sngpt = []
        utils.show_progress("["+str(method)+"] "+"Reading song points... ",i,df.shape[0])
        for j in range(df.shape[1]):
            sngpt.append(df.iloc[i][j])
        sngpts.append(sngpt)

    if(len(vidpts)>len(sngpts)):
        vidpts = vidpts[0:int(len(sngpts))]
    elif(len(sngpts)>len(vidpts)):
        sngpts = sngpts[0:int(len(vidpts))]

    s_vals = []
    s_arss = []
    v_vals = []
    v_arss = []
    p_sims = []

    scltmp1=1.8
    scltmp2=1.8

    for i in range(len(sngpts)):
        s_vals.append(scltmp1*sngpts[i][1]*0.5)
        s_arss.append(scltmp1*sngpts[i][2]*0.5)

    for i in range(len(vidpts)):
        v_vals.append(scltmp2*vidpts[i][1])
        v_arss.append(scltmp2*vidpts[i][2])

    dif_len = abs(len(vidpts)-len(sngpts))
    min_len = min(len(vidpts),len(sngpts))
    max_len = max(len(vidpts),len(sngpts))

    ax_vals.set_xlim(0,max_len)
    ax_arss.set_xlim(0,max_len)
    ax_sims.set_xlim(0,max_len)

    limtmp1 = 1.2

    for i in range(min(len(vidpts),len(sngpts))):
        p_sims.append(combiner.optimizer.calc_similarity_sv(sngpts[i],vidpts[i]))
    m_sim = round(np.mean(p_sims),global_round_numbs)
    d_sim = round( np.std(p_sims),global_round_numbs)

    ax_emos.set_title("2D Emotion Curves (blue=video, green=audio)")
    ax_emos.set_xticks([])
    ax_emos.set_yticks([])
    ax_emos.imshow(imb1)
    ax_emos.set_ylabel("")

    for i in range(len(sngpts)):
        utils.show_progress("["+str(method)+"] "+"Plotting song emotions... ",i,len(sngpts))
        draw_point(ax_emos,s_vals[i],s_arss[i],(0,0.5,0,0.05),"o")
        #if(i<len(p_sims)):
        #    draw_point(ax_emos,s_vals[i],s_arss[i],(1.0,0,0,p_sims[i]**20),"o")
    for i in range(len(vidpts)):
        utils.show_progress("["+str(method)+"] "+"Plotting video emotions... ",i,len(vidpts))
        draw_point(ax_emos,v_vals[i],v_arss[i],(0,0,1.0,0.05),"o")
        #if(i<len(p_sims)):
        #    draw_point(ax_emos,v_vals[i],v_arss[i],(1.0,0,0,p_sims[i]**20),"o")

    ax_vals.set_title("1D Emotion Curves (blue=video, green=audio, red=similarity)")
    ax_vals.set_ylabel("Valence (X)")
    ax_vals.set_xticks([])
    ax_vals.set_ylim([-limtmp1, +limtmp1])
    ax_vals.plot(s_vals,label="audio",color="green")
    ax_vals.plot(v_vals,label="video",color="blue")

    ax_arss.set_ylabel("Arousal (Y)")
    ax_arss.set_xticks([])
    ax_arss.set_ylim([-limtmp1, +limtmp1])
    ax_arss.plot(s_arss,label="audio",color="green")
    ax_arss.plot(v_arss,label="video",color="blue")
    
    ax_sims.set_ylabel("Similarity")
    ax_sims.set_xlabel("Frame id")
    ax_sims.set_ylim([-0.2, +limtmp1])
    ax_sims.plot(p_sims,label="mean: "+str(m_sim),color="red")
    ax_sims.legend(loc='lower left') 

    print("Creating figure... ")

    fntmp1="out/fig_qresbss1x"+str(index)+".png"
    plt.savefig(fntmp1,bbox_inches='tight',dpi=300)
    print("Done! Out file: "+str(fntmp1),)
    plt.show()

def plot_vidpts(index,video_name):

    print("Generating plots...")

    fig = plt.figure(figsize=(20, 10))
    #fig.suptitle(ftit1, fontsize=16)
    grid = plt.GridSpec(2,2)
    grid.update(left=None, bottom=None, right=None, wspace=0.1, hspace=0.1)

    ax_emos = fig.add_subplot(grid[0:2,0:2])
    #ax_vals = fig.add_subplot(grid[0:1,1])
    #ax_arss = fig.add_subplot(grid[1:2,1])

    imb1 = plt.imread("assets/img_back1.png")

    read_fact=1.0

    vidpts_dir = cache_dir+"vidpts/"

    vidpts_csv = vidpts_dir+video_name+"/labels.csv"
    vidpts = []
    df = pd.read_csv(vidpts_csv,header=None)
    for i in range(int(df.shape[0]*read_fact)):
        utils.show_progress("Reading video points... ",i,df.shape[0])
        vidpt = []
        for j in range(df.shape[1]):
            vidpt.append(df.iloc[i][j])
        vidpts.append(vidpt)

    v_vals = []
    v_arss = []

    scltmp2=1.8

    for i in range(len(vidpts)):
        v_vals.append(scltmp2*vidpts[i][1])
        v_arss.append(scltmp2*vidpts[i][2])

    max_len = len(vidpts)

    #ax_vals.set_xlim(0,max_len)
    #ax_arss.set_xlim(0,max_len)

    limtmp1 = 1.2

    #ax_emos.set_title("2D Video Emotion Curve")
    ax_emos.set_xticks([])
    ax_emos.set_yticks([])
    ax_emos.imshow(imb1)
    ax_emos.set_ylabel("")

    for i in range(len(vidpts)):
        utils.show_progress("Plotting video emotions... ",i,len(vidpts))
        draw_point(ax_emos,v_vals[i],v_arss[i],(0,0,1.0,0.05),"o")

    #ax_vals.set_title("1D Video Emotion Curves")
    #ax_vals.set_ylabel("Valence (X)")
    #ax_vals.set_xticks([])
    #ax_vals.set_ylim([-limtmp1, +limtmp1])
    #ax_vals.plot(v_vals,label="video",color="blue")

    #ax_arss.set_ylabel("Arousal (Y)")
    #ax_arss.set_xticks([])
    #ax_arss.set_ylim([-limtmp1, +limtmp1])
    #ax_arss.plot(v_arss,label="video",color="blue")

    print("Creating figure... ")

    fntmp1="out/fig_vidpts1x"+str(index)+".png"
    plt.savefig(fntmp1,bbox_inches='tight',dpi=300)
    print("Done! Out file: "+str(fntmp1),)
    plt.show()

def run():

    mode=3

    if(mode==1):
        vs_pairs = [
            ["Berkeley1","LittleTalks",True],
            ["Berkeley2","OnwardToFreedom",True],
            ["Bike3","InTheEnd",True],
            ["CityWalk1","MyImmortal",True],
            ["MontOldCity1","ThreeDaysGraceLastToKnow",True],
            ["NatureWalk1","LittleTalks",True],
            ["StockHolm1","MyImmortal",True],
            ["Walking4","InTheEnd",True],
        ]

        for i in range(len(vs_pairs)):
            vsp = vs_pairs[i]
            plot_pair(i+1,vsp[0],vsp[1],vsp[2])

    elif(mode==2):

        v_names = [
            "Berkeley1",
            "Berkeley2",
            "Bike3",
            "CityWalk1",
            "MontOldCity1",
            "NatureWalk1",
            "StockHolm1",
            "Walking4",
        ]

        index = int(input("index: "))

        plot_bssres(index,v_names[index-1])
    
    elif(mode==3):

        v_names = [
            "Berkeley1",
            "Berkeley2",
            "Bike3",
            "CityWalk1",
            "MontOldCity1",
            "NatureWalk1",
            "StockHolm1",
            "Walking4",
        ]

        for i in range(1,9):
            index=i
            if(index==8):
                plot_vidpts(index,v_names[index-1])
        
