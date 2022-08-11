#No backups
"""
fine_tunning ref1
class ImageSubnetwork(nn.Module):
    def __init__(self, final_feat_emb_size, pretrained_img_embedder):
        super(ImageSubnetwork, self).__init__()
        resnet50_model = resnet50(pretrained=pretrained_img_embedder)  # pretrained ImageNet ResNet-50
        # Removing the FC layer to add ours!
        modules = list(resnet50_model.children())[:-1]
        self.resnet = nn.Sequential(*modules)
    def forward(self, img):
        img_feats = self.resnet(img)
        return img_feats
    def fine_tune(self, fine_tune=True):
        #Allow or prevent the computation of gradients for (all layers.) #convolutional blocks 2 through 4 of the encoder.
        #:param fine_tune: Allow?
        for p in self.resnet.parameters():
            p.requires_grad = fine_tune
"""

"""
self.model_ft = get_father(father_name)
self.num_ftrs = self.model_ft.fc.in_features
modules = list(self.model_ft.children())[:-1]
self.resnet = nn.Sequential(*modules)
"""

"""
pool = multiprocessing.Pool(4)
out1, out2, out3 = zip(*pool.map(calc_stuff, range(0, 10 * offset, offset))
"""

"""
from header import *

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def get_model(model_name, num_classes, feature_extract=True, use_pretrained=True):

    model_ft = None

    if(model_name=="resnet50"):
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif(model_name=="vgg19"):
        model_ft = models.vgg19(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[3].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif(model_name=="squeezenet1_0"):
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        
        model_ft.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(13)
        )

        model_ft.forward = lambda x: model_ft.classifier(model_ft.features(x)).view(x.size(0), num_classes)
        input_size = 224

    elif(model_name=="custnet01"):
        model_ft = custnet01(num_classes)
        input_size = 128
        
    elif model_name == "resnet":
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size
"""

"""
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
ax1.set_title("Song Profile")
ax1.set_xlabel("Valence")
ax1.set_ylabel("Arousal")
ax1.axis((-1,+1,-1,+1))
ax1.set_title("Song Wave")
def dynamic_plot(i):
    
    #val1=random.randint(-1000,+1000)*0.001
    #val2=random.randint(-1000,+1000)*0.001
    val1=arousals[count[0]]
    val2=valences[count[0]]
    data = np.array([[val2,val1]])
    x, y = data.T
    #ax1.clear()
    
    ax1.plot([+0,+0], [-1,+1], linewidth=0.3,color='red')
    ax1.plot([-1,+1], [+0,+0], linewidth=0.3,color='red')
    ax1.scatter(x,y)
    ax1.set_title("Song Profile ("+str(count[0]/2)+"/"+str(count[1])+"s)"+" Arousal: "+str(val1))
    sys.stdout.write("\033[F")
    sys.stdout.write("\033[K")
    count[0]+=1
    if(count[0]>=len(arousals) or count[0]>=len(valences)):
        count[0]=count[0]-1

def plot_song_profile(song_id):

    deamsong_mp3  = "Songs/"+str(song_id)+".mp3"
    arousals_file = "Labels/arousal.csv"
    valences_file = "Labels/valence.csv"

    print("Loading labels...")

    arousals_str = None
    valences_str = None

    with open(arousals_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if(row[0]==song_id):
                arousals_str=row
                break

    with open(valences_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if(row[0]==song_id):
                valences_str=row
                break

    for i in range(30):
        arousals.append(0)
        valences.append(0)

    for i in range(1,len(arousals_str)):
        arousals.append(float(arousals_str[i]))
    for i in range(1,len(valences_str)):
        valences.append(float(valences_str[i]))

    for i in range(30):
        arousals[i]=arousals[30]
        valences[i]=valences[30]

    count.append(0)
    print("Loading song...")
    
    wavefile = "wave.wav"
    
    sound = AudioSegment.from_mp3(deamsong_mp3)
    sound = sound.set_frame_rate(44100)
    sound.export(wavefile, format="wav")

    spf = wave.open(wavefile,'r')
    signal = spf.readframes(-1)
    signal = np.fromstring(signal,'Int16')

    wave_length = len(AudioSegment.from_file(wavefile))
    count.append(int(wave_length/1000))

    os.remove(wavefile)

    #plt.plot(signal)

    mixer.init()
    mixer.music.load(deamsong_mp3)
    mixer.music.play()
    start_time = time.time()
    
    ani = animation.FuncAnimation(fig,dynamic_plot,interval=500)
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()

count = []
arousals=[]
valences=[]
signal=[]

plot_song_profile(sys.argv[1])
"""

"""
import multiprocessing
from joblib import Parallel, delayed

num_cores = multiprocessing.cpu_count()

class test():
	def foo(self,item,a):
		item = item*item
		print(a)
		return item

	def run(self):
		in_list = []
		for i in range(10):
			in_list.append(i)
		print(in_list)
		
		a = "T"
		out_list = Parallel(n_jobs=num_cores)(delayed(self.foo)(i,a) for i in in_list)
		
		print(out_list)
		
		
test1 = test()
test1.run()
"""

"""
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
"""

"""
def generate_single_image_emojis(i,smt_pnts,img_back1,outdir):
    
    utils.show_progress("Rendering images... ",i,len(smt_pnts))
   
    item = smt_pnts[i]   

    fig1 = plt.figure(figsize=(5,5))
    ax1 = fig1.add_subplot(1,1,1)

    ax1.set_title("Time: "+str(item[0]))
    ax1.set_xlabel("Valence")
    ax1.set_ylabel("Arousal")
    ax1.imshow(img_back1)
    ax1.axis('off')

    if(False):
        draw_point(ax1,item[3],item[4],"red","x")
    
    if(True):
        for j in range(max(i-2,0),i):
            item_j = smt_pnts[j]
            draw_point(ax1,item_j[1],item_j[2],"silver","o")
        draw_point(ax1,item[1],item[2],"red","o")
    
    plt.savefig(outdir+str(i)+".png")

    ax1.clear()
    plt.close()

    return 0
"""


"""
if(model_name=="mernet01"):
    model = mernet01(num_classes)
elif(model_name=="mernet02"):
    model = mernet02(num_classes)
elif(model_name=="resnet50ext"):
    model = resnet50ext(num_classes)
else:
    model = get_other_model(model_name,num_classes)
"""

"""
def calc_similarity_matrix(song_filename,video_filename):

    profgen1 = combiner.profgen.profgen("audio",64,song_filename)
    profgen2 = combiner.profgen.profgen("video", 4,video_filename)

    profgen1.plot_emotion_profile(render_video)
    profgen2.plot_emotion_profile(render_video)

    song_points = profgen1.get_emotion_profile()
    video_points = profgen2.get_emotion_profile()

    song_plen = len(song_points)
    video_plen = len(video_points)

    valence_simmat = np.zeros((song_plen,video_plen))
    arousal_simmat = np.zeros((song_plen,video_plen))

    for spi in range(song_plen):
        song_inst = song_points[spi]
        for vpi in range(video_plen):
            video_inst = video_points[vpi]
            val_sim = abs(song_inst[1]-video_inst[1])
            ars_sim = abs(song_inst[2]-video_inst[2])

            valence_simmat[spi,vpi] = val_sim
            arousal_simmat[spi,vpi] = ars_sim

    plt.matshow(valence_simmat)
    plt.savefig(cache_dir+"valsmtmp1.png")

    plt.matshow(arousal_simmat)
    plt.savefig(cache_dir+"arssmtmp1.png")
"""

"""

    frame_picker_list = []

    xi=0
    xf=0
    rn=0
    while(xf<n_vidfms):
        if(reg_mat[0,xf]!=rn):
            rn=reg_mat[0,xf]
            frame_picker_list.append(frame_picker(xi,xf))
            xi=xf
        xf+=1
    frame_picker_list.append(frame_picker(xi,n_vidfms))

    for fpk in frame_picker_list:
        acc_subvet = acc_mat[0,fpk.xi:fpk.xf]
        fpk.accm = round(np.mean(acc_subvet),2)

    acc_sum = 0
    for fpk in frame_picker_list:
        acc_sum+=fpk.accm
    
    n_drpfms_l = n_drpfms
    for fpk in frame_picker_list:
        fpk.ndrp = round((fpk.accm/acc_sum)*n_drpfms)
        n_drpfms_l-=fpk.ndrp
    print("Drop left: "+str(n_drpfms_l))

    frame_picker_list.sort(key=lambda x: x.accm, reverse=False)

    for fpk in frame_picker_list:
        print(fpk.accm,fpk.ndrp)

"""

"""
-------------------------------------
Algoritmo: Seleção de Melhor Caminho
-------------------------------------
Entrada: Nenhuma
----
Inicialização:
1.	Calcule v=F/M, sendo "v" o speed-up do vídeo, "F" o número de frames do vídeo, e "M" sendo o número de "frames" da música;
2.	Calcule C_v(i,j)=min(||(j-i)-v||^2), a matriz de speed-up para pares de frames;
3.	Calcule a matriz de similaridades para o arousal e o valence, S_a(i,k) e S_v(i,k) respectivamente. //i=1...F e k=1...M

3.	Para i=1 até g faça //onde g é uma janela inicial para salto
		Para j=i+1 até i+w faça //onde w pode ser definido como w=2v, para permitir variados saltos, por exemplo.
			D(i,j,1) = C_v(i,j)
----			
Populando a matriz D:
4.	Para k=2 até M faça
		Para i=g até F faça //F pode ser trocado por k*w, uma vez que não é permitido salto maior que w a cada passo da música 
			Para j=i+1 até i+w faça
				c = C_v(i,j) + \lambda_a / (S_a(i,k-1) + S_a(j, k)) //S_a poderia ser substituído ou usado em conjunto com S_v (Quanto maior o S_a, menor o "c")
				D(i,j,k) = c + min_h[D(i-h,i,k-1)] //Procure o h que minimiza o custo de transição e utilize o valor dado pelo menor
				T(i,j,k) = argmin_h[D(i-h,i,k-1)] //Utilize o h encontrado na equação de cima para definir o melhor caminho para trás		
----
Finalização:
5.	(s,d) = argmin_{i=T-g,j=i+1}^{T,i+w} D(i,j,M)
6.	P = <d>
7.	Enquanto s > g e k > 1 faça
		P = concatene([s,P])
		b = T(s, d, k-1)
		d = s
		s = b
		k = k-1
----
Retorne P
"""

"""
1.	Calcule v=F/M, sendo "v" o speed-up do vídeo, "F" o número de frames do vídeo, e "M" sendo o número de "frames" da música;
2.	Calcule C_v(i,j)=min(||(j-i)-v||^2), a matriz de speed-up para pares de frames;
3.	Calcule a matriz de similaridades para o arousal e o valence, S_a(i,k) e S_v(i,k) respectivamente. //i=1...F e k=1...M
4.  Calcule o vetor de aceleração do vídeo final acc_vet[i] = arousal[i] para i = 1 até M

4. Para i=1 até F faça:
    colocar o frame_i da música na lista de frames da música final
    Para j=i+1 até i+w faça: #w pode ser o número máximo de frames a serem saltados
        calcular os 4 custos, c1,c2,c3,c4
        c1 = custo relativo à aceleração do vídeo para o arousal daquele frame_i da música
        c2 = custo relativo à similaridade inter-frame, dado por C_v
        c3 = custo relativo à similaridade do arousal da música com o arousal do vídeo
        c4 = custo relativo à similaridade do valence da música com o valence do vídeo
        c = c1+c2+c3+c4
        se o frame_j do vídeo tem o menor custo com relação ao frame_i da música, colocá-lo na lista de frames do vídeo final

retornar as listas de frames da música e do vídeo finais
"""

"""
class line():
    def __init__(self,xi,xf,y):
        self.xi = xi
        self.xf = xf
        self.y = y

        self.w = self.xf+1-self.xi
        self.h = 1

class rect():
    def __init__(self,xi,xf,yi,yf):
        self.xi = xi
        self.xf = xf
        self.yi = yi
        self.yf = yf
        self.line_list = []
        self.accm=0
        self.nf_tot=0
        self.nf_drq=0
        self.nf_dgv=0
        self.nf_sgv=0

        self.w = self.xf+1-self.xi
        self.h = self.yf+1-self.yi

    def draw_lines(self):
        for lin in self.line_list:
            print("  line",lin.xi,lin.xf,lin.y)
    
    def draw_values(self):
        print("  Frames: "+str(self.nf_tot))
        print("  To drop: "+str(self.nf_dgv))
        print("  To keep: "+str(self.nf_sgv))

    def draw(self):
        print("rect",self.xi,self.xf,self.yi,self.yf)
        self.draw_values()
        self.draw_lines()

def reduce_video_distributed(song_points,video_points):

    n_sngfms = len(song_points)
    n_vidfms = len(video_points)

    sim_mat = np.zeros((n_sngfms,n_vidfms))
    lin_mat = np.zeros((n_sngfms,n_vidfms))
    rec_mat = np.zeros((n_sngfms,n_vidfms))
    lrc_mat = np.zeros((n_sngfms,n_vidfms))
    sel_mat = np.zeros((n_sngfms,n_vidfms))

    for i in range(n_sngfms):
        song_point = song_points[i]
        utils.show_progress("Calculating similarity...",i,n_sngfms)
        for j in range(n_vidfms):
            video_point = video_points[j]
            sim_mat[i,j] = calc_similarity(song_point,video_point)
    
    acc_vet = []
    for i in range(len(song_points)):
        song_point = song_points[i]
        acc = song_point[2]
        acc = (acc+1)/2
        acc_vet.append(acc) 

    cv2.line(lin_mat, (0, 0), (n_vidfms-1, n_sngfms-1), (1, 1, 1), thickness=1)

    line_list = []

    xi=0
    xf=0
    y=0
    while(xf<n_vidfms):
        if(lin_mat[y,xf]==0):
            line_list.append(line(xi,xf-1,y))
            xi=xf
            y+=1
        xf+=1
    line_list.append(line(xi,n_vidfms-1,y))

    for lin in line_list:
        lin_mat[lin.y,lin.xi:lin.xf+1] = lin.y+1

    rect_list = []

    rec_h = int(n_vidfms/n_sngfms)
    for i in range(0,len(line_list),rec_h):
        line_i = line_list[i]
        line_f = line_list[min(i+rec_h-1,len(line_list)-1)]
        xi=line_i.xi
        xf=line_f.xf
        yi=line_i.y
        yf=line_f.y
        rec1 = rect(xi,xf,yi,yf)
        rect_list.append(rec1)
        for k in range(i,min(i+rec_h,len(line_list))):
            rec1.line_list.append(line_list[k])

    for rec in rect_list:
        rec_mat[rec.yi:rec.yf+1,rec.xi:rec.xf+1] = 1

    nf_dgv_tot = n_vidfms - n_sngfms
    nf_drq_tot = 0

    for rec in rect_list:
        acc_vet_i = acc_vet[rec.yi:rec.yf+1]
        rec.accm = np.mean(acc_vet_i)
        rec.nf_tot = rec.w
        rec.nf_drq = int(round((rec.w-1)*rec.accm))
        nf_drq_tot+=rec.nf_drq
    
    #print("Need to drop",nf_dgv_tot)
    #print("Rqst to drop",nf_drq_tot)

    rggv_rat = nf_dgv_tot/nf_drq_tot
    
    for rec in rect_list:
        rec.nf_dgv = int(round(rec.nf_drq*rggv_rat))

    #Redistribution method
    for k in range(10):
        for i in range(len(rect_list)):
            rec_i = rect_list[i]
            if(rec_i.nf_dgv>rec_i.nf_tot-1):
                j=i-1
                while(j>0):
                    rec_j = rect_list[j]
                    if(rec_j.nf_dgv<rec_j.nf_tot-2):
                        rec_i.nf_dgv-=1
                        rec_j.nf_dgv+=1
                        break
                    j-=1
            
            if(rec_i.nf_dgv>rec_i.nf_tot-1):
                j=i+1
                while(j<len(rect_list)):
                    rec_j = rect_list[j]
                    if(rec_j.nf_dgv<rec_j.nf_tot-2):
                        rec_i.nf_dgv-=1
                        rec_j.nf_dgv+=1
                        break
                    j+=1

    for rec in rect_list:
        rec.nf_dgv = min(rec.nf_dgv,rec.nf_tot-1)
        rec.nf_sgv = rec.nf_tot-rec.nf_dgv

    nf_dgv_tot_n=0
    nf_sgv_tot_n=0
    for rec in rect_list:
        nf_dgv_tot_n+=rec.nf_dgv
        nf_sgv_tot_n+=rec.nf_sgv
    nf_dgv_left = nf_dgv_tot-nf_dgv_tot_n

    ovfws=0
    for rec in rect_list:
        if(rec.nf_dgv>rec.nf_tot-1):
            ovfws+=1

    #print("Total to drop:",nf_dgv_tot)
    #print("Given to drop:",nf_dgv_tot_n)
    #print("Given to keep:",nf_sgv_tot_n)
    #print("To drop left:",nf_dgv_left)
    #print("Acc overflows:",ovfws)

    for rec in rect_list:
        lrc_mat[rec.yi:rec.yf+1,rec.xi:rec.xf+1] = 1
        for lin in rec.line_list:
            lrc_mat[lin.y,lin.xi:lin.xf+1] = 2
        sel_mat[:,rec.xi:rec.xi+rec.nf_sgv] = 1

    video_points_f = []
    song_points_f = song_points

    for x in range(n_vidfms):
        if(sel_mat[0,x]==1):
            video_points_f.append(video_points[x])

    print("Song/video lengths: ",len(song_points_f),len(video_points_f))

    n_pnts_left = len(video_points_f)-len(song_points_f)
    video_points_f2 = []
    i=0
    k=0
    if(n_pnts_left<0):
        ks=int(-len(video_points_f)/n_pnts_left)
        while(i<len(video_points_f)):
            video_points_f2.append(video_points_f[i])
            if(k>=ks):
                k=0
                video_points_f2.append(video_points_f[i])
            i+=1
            k+=1
    else:
        ks=int(len(video_points_f)/n_pnts_left)
        while(i<len(video_points_f)):
            if(k>=ks):
                k=0
            else:
                video_points_f2.append(video_points_f[i])
            i+=1
            k+=1

    video_points_f = video_points_f2

    print("Song/video lengths: ",len(song_points_f),len(video_points_f))

    n_pnts_left = len(video_points_f)-len(song_points_f)
    if(n_pnts_left>0):
        video_points_f = video_points_f[0:int(len(song_points_f))]
    else:
        for i in range(len(video_points_f),len(song_points_f)):
            video_points_f.append(video_points_f[len(video_points_f)-1])

    print("Song/video lengths: ",len(song_points_f),len(video_points_f))

    plot_matrix(sim_mat)
    plot_matrix(lin_mat)
    plot_matrix(rec_mat)
    plot_matrix(lrc_mat)
    plot_matrix(sel_mat)

    return song_points_f,video_points_f

def reduce_video_distributed_2(song_points,video_points):

    n_sngfms = len(song_points)
    n_vidfms = len(video_points)

    sim_mat = np.zeros((n_sngfms,n_vidfms))
    lin_mat = np.zeros((n_sngfms,n_vidfms))
    rec_mat = np.zeros((n_sngfms,n_vidfms))
    lrc_mat = np.zeros((n_sngfms,n_vidfms))
    sel_mat = np.zeros((n_sngfms,n_vidfms))


    for i in range(n_sngfms):
        song_point = song_points[i]
        utils.show_progress("Calculating similarity...",i,n_sngfms)
        for j in range(n_vidfms):
            video_point = video_points[j]
            sim_mat[i,j] = calc_similarity(song_point,video_point) 

    rec_h = 8
    acc_msk = [8,8,4,4,2,2,1,1]

    acc_vet = []
    for i in range(len(song_points)):
        song_point = song_points[i]
        acc = song_point[2]
        acc = (acc+1)/2
        acc = int(acc*7)
        acc_vet.append(acc) 

    cv2.line(lin_mat, (0, 0), (n_vidfms-1, n_sngfms-1), (1, 1, 1), thickness=1)

    line_list = []

    xi=0
    xf=0
    y=0
    while(xf<n_vidfms):
        if(lin_mat[y,xf]==0):
            line_list.append(line(xi,xf-1,y))
            xi=xf
            y+=1
        xf+=1
    line_list.append(line(xi,n_vidfms-1,y))

    for lin in line_list:
        lin_mat[lin.y,lin.xi:lin.xf+1] = lin.y+1

    rect_list = []

    for i in range(0,len(line_list),rec_h):
        line_i = line_list[i]
        line_f = line_list[min(i+rec_h-1,len(line_list)-1)]
        xi=line_i.xi
        xf=line_f.xf
        yi=line_i.y
        yf=line_f.y
        rec1 = rect(xi,xf,yi,yf)
        rect_list.append(rec1)
        for k in range(i,min(i+rec_h,len(line_list))):
            rec1.line_list.append(line_list[k])

    for rec in rect_list:
        rec_mat[rec.yi:rec.yf+1,rec.xi:rec.xf+1] = 1

    for rec in rect_list:
        acc_vet_i = acc_vet[rec.yi:rec.yf+1]
        rec.accm = int(np.mean(acc_vet_i))
        step = acc_msk[rec.accm]
        for i in range(0,len(rec.line_list),step):
            xfm = rec.line_list[i].xi
            yfm = rec.line_list[i].y
            sel_mat[:,xfm] = 2
            sel_mat[:,xfm+1:xfm+step] = 1

    plot_matrix(sel_mat,False)

    video_points_f = []
    song_points_f = song_points

    for x in range(n_vidfms):
        if(sel_mat[0,x]==2):
            video_points_f.append(video_points[x])
            x2=x+1
            while(sel_mat[0,x2]==1):
                video_points_f.append(video_points[x2])
                x2+=1
                if(x2>=n_vidfms):
                    break

    print("Song/video lengths: ",len(song_points_f),len(video_points_f))


    #Distribute left frames uniform in video
    n_pnts_left = len(video_points_f)-len(song_points_f)
    video_points_f2 = []
    i=0
    k=0
    if(n_pnts_left<0):
        ks=int(-len(video_points_f)/n_pnts_left)
        while(i<len(video_points_f)):
            video_points_f2.append(video_points_f[i])
            if(k>=ks):
                k=0
                video_points_f2.append(video_points_f[i])
            i+=1
            k+=1
    else:
        ks=int(len(video_points_f)/n_pnts_left)
        while(i<len(video_points_f)):
            if(k>=ks):
                k=0
            else:
                video_points_f2.append(video_points_f[i])
            i+=1
            k+=1
    video_points_f = video_points_f2
    print("Song/video lengths: ",len(song_points_f),len(video_points_f))

    #Adjust left frames at end
    n_pnts_left = len(video_points_f)-len(song_points_f)
    if(n_pnts_left>0):
        video_points_f = video_points_f[0:int(len(song_points_f))]
    else:
        for i in range(len(video_points_f),len(song_points_f)):
            video_points_f.append(video_points_f[len(video_points_f)-1])

    print("Song/video lengths: ",len(song_points_f),len(video_points_f))

    return song_points_f,video_points_f
"""

"""
def reduce_video_sequential_optimized(song_points,video_points):

    n_sngfms = len(song_points)
    n_vidfms = len(video_points)
    n_vsdiff = n_vidfms/n_sngfms

    acc_vet = []
    for i in range(len(song_points)):
        song_point = song_points[i]
        acc = song_point[2]
        acc = (acc+1)/2
        acc = int(9*acc)+1
        acc_vet.append(acc)

    acc_vet2 = []
    i_s=0
    i_v=0
    while(i_s<len(song_points)):
        for j in range(int(i_v),int(i_v+n_vsdiff)):
            acc_vet2.append(acc_vet[i_s])
        i_s+=1
        i_v+=n_vsdiff

    acc_vet = acc_vet2
        
    video_points_f = []
    song_points_f = song_points

    error = 1
    max_error = 0.01

    acc_ref1i = 1
    acc_ref1m = 20
    acc_ref1n = 0
    acc_ref1s = 0.00001

    acc_max1i = 5
    acc_max1m = 20
    acc_max1n = 1
    acc_max1s = 0.00001

    max_i=1e6
    fail=False
    while(True):
        max_i+=1
        i=0
        while(i<len(video_points)):
            video_points_f.append(video_points[int(i)])
            k=acc_vet[int(i)]*acc_ref1i
            i+=k

        vidsng_diff = len(video_points_f)-len(song_points_f)
        error = vidsng_diff/len(video_points_f)

        if(abs(error)>max_error):
            if(error<0):
                acc_ref1i-=acc_ref1s
                if(acc_ref1i<=acc_ref1n):
                    fail=True
                    break
            else:
                acc_ref1i+=acc_ref1s
                if(acc_ref1i>=acc_ref1m):
                    fail=True
                    break
        else:
            break
        
        if(i>max_i):
            fail=True
            break

        utils.wait()

        print("sf:",len(song_points_f),"vf:",len(video_points_f),"df:",vidsng_diff,"er:",error)

    if(fail==True):
        print("Failed to find best fit!")
        exit()

    print("Song/video lengths: ",len(song_points_f),len(video_points_f))

    #Distribute left frames uniform in video
    vidsng_diff = len(video_points_f)-len(song_points_f)
    video_points_f2 = []
    i=0
    k=0
    if(vidsng_diff<0):
        ks=int(-len(video_points_f)/vidsng_diff)
        while(i<len(video_points_f)):
            video_points_f2.append(video_points_f[i])
            if(k>=ks):
                k=0
                video_points_f2.append(video_points_f[i])
            i+=1
            k+=1
    else:
        ks=int(len(video_points_f)/vidsng_diff)
        while(i<len(video_points_f)):
            if(k>=ks):
                k=0
            else:
                video_points_f2.append(video_points_f[i])
            i+=1
            k+=1
    video_points_f = video_points_f2
    print("Song/video lengths: ",len(song_points_f),len(video_points_f))

    #Adjust left frames at end
    vidsng_diff = len(video_points_f)-len(song_points_f)
    if(vidsng_diff>0):
        video_points_f = video_points_f[0:int(len(song_points_f))]
    else:
        for i in range(len(video_points_f),len(song_points_f)):
            video_points_f.append(video_points_f[len(video_points_f)-1])
    print("Song/video lengths: ",len(song_points_f),len(video_points_f))

    for item in video_points_f:
        print(item)

    return song_points_f,video_points_f
"""

"""
-------------------------------------
Algoritmo: Seleção de Melhor Caminho
-------------------------------------
Entrada: Nenhuma
----
Inicialização:
1.	Calcule v=F/M, sendo "v" o speed-up do vídeo, "F" o número de frames do vídeo, e "M" sendo o número de "frames" da música;
2.	Calcule C_v(i,j)=min(||(j-i)-v||^2), a matriz de speed-up para pares de frames;
3.	Calcule a matriz de similaridades para o arousal e o valence, S_a(i,k) e S_v(i,k) respectivamente. //i=1...F e k=1...M

4.	Para i=1 até g faça //onde g é uma janela inicial para salto
        Para j=i+1 até i+w faça //onde w pode ser definido como w=2v, para permitir variados saltos, por exemplo.
            D(i,j,1) = C_v(i,j)
----			
Populando a matriz D:
5.	Para k=2 até M faça
        Para i=g até F faça //F pode ser trocado por k*w, uma vez que não é permitido salto maior que w a cada passo da música 
            Para j=i+1 até i+w faça
                c = C_v(i,j) + \lambda_a / (S_a(i,k-1) + S_a(j, k)) //S_a poderia ser substituído ou usado em conjunto com S_v (Quanto maior o S_a, menor o "c")
                D(i,j,k) = c + min_h[D(i-h,i,k-1)] //Procure o h que minimiza o custo de transição e utilize o valor dado pelo menor
                T(i,j,k) = argmin_h[D(i-h,i,k-1)] //Utilize o h encontrado na equação de cima para definir o melhor caminho para trás		
----
Finalização:
6.	(s,d) = argmin_{i=T-g,j=i+1}^{T,i+w} D(i,j,M)
7.	P = <d>
8.	Enquanto s > g e k > 1 faça
        P = concatene([s,P])
        b = T(s, d, k-1)
        d = s
        s = b
        k = k-1
----
Retorne P
"""

"""
def reduce_video_dinprog(song_points,video_points):

    video_points = video_points[:2000]
    song_points = song_points[:200]

    print("Song/video lengths: ",len(song_points),len(video_points))

    M = len(song_points)
    F = len(video_points)
    v = F/M

    C_v = np.zeros((F,F))
    S_v = np.zeros((F,M))
    S_a = np.zeros((F,M))

    for i in range(M):
        utils.show_progress("Computing C_v... ",i,M)
        for j in range(M):
            C_v[i,j] = min(((j-i)-v)**2,200)

    for i in range(F):
        utils.show_progress("Computing S_v and S_a... ",i,F)
        for j in range(M):
            S_v[i,j] = 0.5*abs(video_points[i][1]-song_points[j][1])
            S_a[i,j] = 0.5*abs(video_points[i][2]-song_points[j][2])

    print(C_v)
    utils.genaral.wait()
    print(S_v)
    utils.genaral.wait()
    print(S_a)
    utils.genaral.wait()

    g = 4
    w = int(2*v)

    D = np.zeros((F,F,M),dtype=np.float32)
    T = np.zeros((F,F,M),dtype=np.float32)

    for i in range(1,g):
        for j in range(i+1,i+w):
            D[i,j,1] = C_v[i,j]

    lambda_a = 1

    for k in range(2,M):
        utils.show_progress("Computing D and T... ",k,M)
        for i in range(g,F):
            for j in range(i+1,min(i+w,M)):
                c = C_v[i,j] + lambda_a/(S_a[i,k-1] + S_a[j,k])
                min_h = D[i,i,k-1]
                argmin_h = i
                i2=i
                while(i2>-1):
                    if(D[i2,i,k-1]<min_h):
                        min_h = D[i2,i,k-1]
                        argmin_h = i2
                    i2-=1

                D[i,j,k] = c + min_h
                T[i,j,k] = c + argmin_h

    print(D)
    print(T)

    utils.wait()

    s = F-g
    d = s+1
    min_sd = D[s,d,M-1]

    for i in range(F-g,F):
        for j in range(i+1,min(i+w,M)):
            if(D[i,j,M]<min_sd):
                min_sd = D[i,j,M]
                s = i
                d = j
    P = []
    P.append(d)

    print(P)
    utils.wait()

    while(s>g and k>1):
        P.append(s)
        b = T[s,d,k-1]
        d = s
        s = b
        k = k-1

    P.reverse()

    print(P)

    video_points_f = []
    song_points_f = []

    for item in P:
        video_points_f.append(video_points[item])
    
    for i in range(len(video_points_f)):
        song_points_f.append(song_points[i])

    return song_points_f, video_points_f
"""

"""
#Best version of optimizer_dinprog
class optimizer_dinprog():

    def comput_mat(self,type,i,F,M,v,vp,sp):
        if(type==1):
            ltmp1 = [0]*F
            utils.show_progress("Computing C_s... ",i,F)
            for j in range(F):
                ltmp1[j] = min(((j-i)-v)**2,200)
        elif(type==2):
            ltmp1 = [0]*F
            utils.show_progress("Computing S_i... ",i,F)
            for j in range(max(i-v,0),min(i+v,F)):
                ltmp1[j] = calc_similarity_vv(vp[i],vp[j])
        elif(type==3):
            ltmp1 = [0]*M
            utils.show_progress("Computing S_v... ",i,F)
            for j in range(M):
                ltmp1[j] = 1 - 0.5*abs(vp[i][1]-sp[j][1])
        elif(type==4):
            ltmp1 = [0]*M
            utils.show_progress("Computing S_a... ",i,F)
            for j in range(M):
                ltmp1[j] = 1 - 0.5*abs(vp[i][2]-sp[j][2])
        
        return ltmp1

    def optimize(self,song_points,video_points):

        M = len(song_points)
        F = len(video_points)
        v = int(F/M)

        C_s = np.zeros((F,F))
        S_i = np.zeros((F,F))
        S_v = np.zeros((F,M))
        S_a = np.zeros((F,M))

        ltmp1 = Parallel(n_jobs=num_cores)(
            delayed(self.comput_mat)(1,i,F,M,v,video_points,song_points)
                for i in range(F))
        for i in range(F):
            for j in range(F):
                C_s[i,j] = ltmp1[i][j]

        ltmp1 = Parallel(n_jobs=num_cores)(
            delayed(self.comput_mat)(2,i,F,M,v,video_points,song_points)
                for i in range(F))
        for i in range(F):
            for j in range(max(i-v,0),min(i+v,F)):
                S_i[i,j] = ltmp1[i][j]

        #ltmp1 = Parallel(n_jobs=num_cores)(
        #    delayed(self.comput_mat)(3,i,F,M,v,video_points,song_points)
        #        for i in range(F))
        #for i in range(F):
        #    for j in range(M):
        #        S_v[i,j] = ltmp1[i][j]

        #ltmp1 = Parallel(n_jobs=num_cores)(
        #    delayed(self.comput_mat)(4,i,F,M,v,video_points,song_points)
        #        for i in range(F))
        #for i in range(F):
        #    for j in range(M):
        #        S_a[i,j] = ltmp1[i][j]

        for i in range(F):
            utils.show_progress("Computing S_v and S_a... ",i,F)
            for j in range(M):
                S_v[i,j] = 1 - 0.5*abs(video_points[i][1]-song_points[j][1])
                S_a[i,j] = 1 - 0.5*abs(video_points[i][2]-song_points[j][2])    

        g = 2
        w = int(4*v)

        D = np.full((F,F,M),float('inf'))
        T = np.full((F,F,M),float('nan'))

        for i in range(F):
            for j in range(i+1,min(i+w,F)):
                D[i,j,0] = C_s[i,j]

        p_totspd = 0
        p_arsspd = 0
        p_frmsim = 0
        p_valsim = 0.5
        p_arssim = 0.5

        for k in range(1,M):
            utils.show_progress("Computing D and T... ",k-1,M-1)
            for i in range(g,F):

                h=1
                min_h = D[i-h,i,k-1]
                argmin_h = i-h
                while(i-h>-1):
                    if(D[i-h,i,k-1]<min_h):
                        min_h = D[i-h,i,k-1]
                        argmin_h = i-h
                    h+=1

                for j in range(i+1,min(i+w,F)):

                    c_totspd = C_s[i,j]     #Custo do speedup total
                    c_frmsim = 1-S_i[i,j]   #Custo da similaridade interframes
                    c_arsspd = 0            #Custo da aceleração por arousal
                    c_valsim = 1-S_v[j,k]   #Custo da similaridade por valence
                    c_arssim = 1-S_a[j,k]   #Custo da similaridade por arousal

                    c = p_totspd*c_totspd + p_frmsim*c_frmsim + p_arsspd*c_arsspd + p_valsim*c_valsim + p_arssim*c_arssim

                    D[i,j,k] = c + min_h
                    T[i,j,k] = argmin_h

        s = F-g
        d = s+1
        min_sd = D[s,d,M-1]

        for i in range(F-g,F):
            for j in range(i+1,min(i+w,M)):
                if(D[i,j,M-1]<min_sd):
                    min_sd = D[i,j,M-1]
                    s = i
                    d = j
        P = []
        P.append(d)
        k=M

        while(s>g and k>1):
            P.append(int(s))
            b = T[int(s),int(d),int(k-1)]
            d = int(s)
            s = int(b)
            k = k-1

        P.reverse()

        video_points_f = []
        song_points_f = []

        for item in P:
            video_points_f.append(video_points[item])
        
        for i in range(len(video_points_f)):
            song_points_f.append(song_points[i])

        return song_points_f, video_points_f
    """

    """
    #Sparse version of optimizer dinprog
    class optimizer_dinprog():

    def show_mat(self,D,T):

        print(D)
        n_infs = 0
        n_nmbs = 0
        for i in range(F):
            for j in range(F):
                for k in range(M):
                    if(math.isinf(D[i,j,k])):
                        n_infs+=1
                    else:
                        n_nmbs+=1
        print("n_infs: "+str(n_infs))
        print("n_nmbs: "+str(n_nmbs))

        print(T)
        n_nans = 0
        n_nmbs = 0
        for i in range(F):
            for j in range(F):
                for k in range(M):
                    if(math.isnan(T[i,j,k])):
                        n_nans+=1
                    else:
                        n_nmbs+=1
        print("n_nans: "+str(n_nans))
        print("n_nmbs: "+str(n_nmbs))

    def comput_mat(self,type,i,F,M,v,vp,sp):
        if(type==1):
            ltmp1 = [0]*F
            utils.show_progress("Computing C_s... ",i,F)
            for j in range(F):
                ltmp1[j] = min(((j-i)-v)**2,200)
        elif(type==2):
            ltmp1 = [0]*F
            utils.show_progress("Computing S_i... ",i,F)
            for j in range(max(i-v,0),min(i+v,F)):
                ltmp1[j] = calc_similarity_vv(vp[i],vp[j])
        elif(type==3):
            ltmp1 = [0]*M
            utils.show_progress("Computing S_v... ",i,F)
            for j in range(M):
                ltmp1[j] = 1 - 0.5*abs(vp[i][1]-sp[j][1])
        elif(type==4):
            ltmp1 = [0]*M
            utils.show_progress("Computing S_a... ",i,F)
            for j in range(M):
                ltmp1[j] = 1 - 0.5*abs(vp[i][2]-sp[j][2])
        
        return ltmp1

    def optimize(self,song_points,video_points):

        M = len(song_points)
        F = len(video_points)
        v = int(F/M)

        C_s = np.zeros((F,F))
        S_i = np.zeros((F,F))
        S_v = np.zeros((F,M))
        S_a = np.zeros((F,M))
        
        #ltmp1 = Parallel(n_jobs=num_cores)(
        #    delayed(self.comput_mat)(1,i,F,M,v,video_points,song_points)
        #        for i in range(F))
        #for i in range(F):
        #    for j in range(F):
        #        C_s[i,j] = ltmp1[i][j]

        #ltmp1 = Parallel(n_jobs=num_cores)(
        #    delayed(self.comput_mat)(2,i,F,M,v,video_points,song_points)
        #        for i in range(F))
        #for i in range(F):
        #    for j in range(max(i-v,0),min(i+v,F)):
        #        S_i[i,j] = ltmp1[i][j]

        #ltmp1 = Parallel(n_jobs=num_cores)(
        #    delayed(self.comput_mat)(3,i,F,M,v,video_points,song_points)
        #        for i in range(F))
        #for i in range(F):
        #    for j in range(M):
        #        S_v[i,j] = ltmp1[i][j]

        #ltmp1 = Parallel(n_jobs=num_cores)(
        #    delayed(self.comput_mat)(4,i,F,M,v,video_points,song_points)
        #        for i in range(F))
        #for i in range(F):
        #    for j in range(M):
        #        S_a[i,j] = ltmp1[i][j]

        for i in range(F):
            utils.show_progress("Computing C_s... ",i,F)
            for j in range(F):
                C_s[i,j] = min(((j-i)-v)**2,200)
        
        #for i in range(F):
        #    utils.show_progress("Computing S_i... ",i,F)
        #    for j in range(max(i-v,0),min(i+v,F)):
        #        S_i[i,j] = calc_similarity_vv(video_points[i],video_points[j])

        for i in range(F):
            utils.show_progress("Computing S_v and S_a... ",i,F)
            for j in range(M):
                S_v[i,j] = 1 - 0.5*abs(video_points[i][1]-song_points[j][1])
                S_a[i,j] = 1 - 0.5*abs(video_points[i][2]-song_points[j][2])

        g = 2
        w = int(1*v)

        D = [0]*M
        T = [0]*M

        D[0] = csr_matrix(np.full((F,F),float('inf')))
        T[0] = csr_matrix(np.full((F,F),float('nan')))

        for i in range(F):
            utils.show_progress("Creating D[0]... ",i,F)
            for j in range(i+1,min(i+w,F)):
                D[0][i,j] = C_s[i,j]

        p_totspd = 0
        p_arsspd = 0
        p_frmsim = 0
        p_valsim = 0.5
        p_arssim = 0.5

        for k in range(1,M):
            utils.show_progress("Computing D and T... ",k-1,M-1)
            Dk = np.full((F,F),float('inf'))
            Tk = np.full((F,F),float('nan'))
            Dk1 = D[k-1].toarray()
            for i in range(g,F):
                for j in range(i+1,min(i+w,F)):

                    c_totspd = C_s[i,j]     #Custo do speedup total
                    c_frmsim = 1-S_i[i,j]   #Custo da similaridade interframes
                    c_arsspd = 0            #Custo da aceleração por arousal
                    c_valsim = 1-S_v[j,k]   #Custo da similaridade por valence
                    c_arssim = 1-S_a[j,k]   #Custo da similaridade por arousal

                    c = p_totspd*c_totspd + p_frmsim*c_frmsim + p_arsspd*c_arsspd + p_valsim*c_valsim + p_arssim*c_arssim

                    h=1
                    min_h = Dk1[i-h,i]
                    argmin_h = i-h
                    while(i-h>-1):
                        if(Dk1[i-h,i]<min_h):
                            min_h = Dk1[i-h,i]
                            argmin_h = i-h
                        h+=1

                    Dk[i,j] = c + min_h
                    Tk[i,j] = argmin_h

            D[k] = csr_matrix(Dk)
            T[k] = csr_matrix(Tk)
            Dk = None
            Tk = None
            Dk1 = None

        s = F-g
        d = s+1
        min_sd = D[M-1][s,d]

        for i in range(F-g,F):
            for j in range(i+1,min(i+w,M)):
                if(D[M-1][i,j]<min_sd):
                    min_sd = D[M-1][i,j]
                    s = i
                    d = j
        P = []
        P.append(d)
        k=M

        while(s>g and k>1):
            P.append(int(s))
            b = T[int(k-1)][int(s),int(d)]
            d = int(s)
            s = int(b)
            k = k-1

        P.reverse()

        video_points_f = []
        song_points_f = []

        for item in P:
            video_points_f.append(video_points[item])
        
        for i in range(len(video_points_f)):
            song_points_f.append(song_points[i])

        return song_points_f, video_points_f
    """

    """
    #Trying to improve min calculations
    for k in range(M):

            utils.show_progress("Computing D and T... ",k,M)
            D[k] = sparse_matrix(F,float('inf'))
            T[k] = sparse_matrix(F,float('nan'))

            #mhk = np.full((F,F),float('inf'))
            #ahk = np.full((F,F),float('nan'))
            #mhv = np.full((F),float('inf'))
            #ahv = np.full((F),float('nan'))

            #h=1
            #min_h = D[i-h,i,k-1]
            #argmin_h = i-h
            #while(i-h>=0):
            #    if(D[i-h,i,k-1]<min_h):
            #        min_h = D[i-h,i,k-1]
            #        argmin_h = i-h
            #    h+=1

            if(k==0):
                for i in range(F):
                    for j in range(i+1,min(i+w,F)):
                        D[k].set_val(i,j,C_s[i,j])

                        #vtmp1 = C_s[i,j]
                        #if(vtmp1<mhv[j]):
                        #    mhv[j] = vtmp1
                        #    ahv[j] = i
                        #mhk[i,j] = mhv[j]
                        #ahk[i,j] = ahv[j]
                        
            else:                
                for i in range(g,min(k*w+1,F)):

                    min_h, argmin_h = D[k-1].get_min(i)
                    #min_h, argmin_h = mhk[i-1,i], ahk[i-1,i]

                    for j in range(i+1,min(i+w,F)):                  

                        c_totspd = C_s[i,j]     #Custo do speedup total
                        c_frmsim = 1-S_i[i,j]   #Custo da similaridade interframes
                        c_arsspd = 0            #Custo da aceleração por arousal
                        c_valsim = 1-S_v[j,k]   #Custo da similaridade por valence
                        c_arssim = 1-S_a[j,k]   #Custo da similaridade por arousal

                        c = p_totspd*c_totspd + p_frmsim*c_frmsim + p_arsspd*c_arsspd + p_valsim*c_valsim + p_arssim*c_arssim
                            
                        D[k].set_val(i,j,c+min_h)
                        T[k].set_val(i,j,argmin_h)

                        #vtmp1 = c+min_h
                        #if(vtmp1<mhv[j]):
                        #    mhv[j] = vtmp1
                        #    ahv[j] = i
                        #mhk[i,j] = mhv[j]
                        #ahk[i,j] = ahv[j]
    """

    """
    def comput_slc_p(self,i,F,w):

        i1,i2,k,n = i[0],i[1],i[2],i[3]

        Mkvs = []

        #print("  Init part "+str(k)+"/"+str(n))

        for i in range(i1,i2):

            min_h, argmin_h = self.Dk1.get_min(i)

            Dkiv = []
            Tkiv = []

            for j in range(i+1,min(i+1+w,F)):

                c_totspd = C_s[i,j]     #Custo do speedup total
                c_arsspd = C_a[i,j]     #Custo da aceleração por arousal
                c_frmsim = 1-S_i[i,j]   #Custo da similaridade interframes
                c_valsim = 1-S_v[j,k]   #Custo da similaridade por valence
                c_arssim = 1-S_a[j,k]   #Custo da similaridade por arousal

                c = self.p_totspd*c_totspd + self.p_frmsim*c_frmsim + self.p_arsspd*c_arsspd + self.p_valsim*c_valsim + self.p_arssim*c_arssim

                Dkiv.append([j,c+min_h])
                Tkiv.append([j,argmin_h])    

            Mkvs.append([Dkiv,Tkiv])

        #print("  Done part "+str(k)+"/"+str(n))
        
        return Mkvs
    """

    """
    #Parallel version (worst)

    self.Dk1 = D[k-1]

    njobs=4
    ljobs = utils.get_ljobs(g,min(k*w+1,F),njobs)
    louts = Parallel(n_jobs=njobs)(
        delayed(self.comput_slc_p)(i,F,w)
            for i in ljobs)

    Dkiv=[]
    Tkiv=[]
    for jbi in range(len(ljobs)):
        #utils.show_progress("Filling D["+str(k)+"/"+str(M)+"]... ",jbi,len(ljobs))
        i1,i2 = ljobs[jbi][0],ljobs[jbi][1]
        for i in range(i1,i2):        
            D[k].set_val_v(i,louts[jbi][i-i1][0])
            T[k].set_val_v(i,louts[jbi][i-i1][1])
    """

"""
#move_files
import os

main_dir = "/srv/storage/datasets/Diognei/Audio/DEAM/songs/"

k_i = 0
k_f = 2060

for k in range(k_i,k_f):
    print(str(k)+"/"+str(k_f))
    song_dir = main_dir+str(k)+"/"
    if(os.path.exists(song_dir)):
        frames_dir=song_dir+"frames/"
        if not(os.path.exists(frames_dir)):
            os.mkdir(frames_dir)
        for root, dirs, files in os.walk(song_dir):
            for file in files:
                if(file.endswith(".csv") or file.endswith(".wav")):
                    if(os.path.exists(song_dir+file)):
                        os.rename(song_dir+file,frames_dir+file)
"""

"""
class optimizer_dtw():

    def optimize(self,song_points,video_points):
        
        xlist = []
        for i in song_points:
            xlist.append([i[1]+i[2]])
        ylist = []
        for i in video_points:
            ylist.append([i[1]+i[2]])
        
        x = np.array(xlist)
        y = np.array(ylist)

        manhattan_distance = lambda x, y: np.abs(x - y)

        d, cost_matrix, acc_cost_matrix, path = dtw(x, y, dist=manhattan_distance)
                
        song_points_f = []
        video_points_f = []

        for i in range(len(path[0])-1):
            if (path[0][i] != path[0][i+1]):
                song_points_f.append(song_points[path[0][i]])
                video_points_f.append(video_points[path[1][i]])

        return song_points_f, video_points_f
"""

"""
#Project song arousal in video
lsp = len(song_points)
lvp = len(video_points)
div = lsp/lvp
arsp_vet = [0]*lvp
for i_v in range(lvp):
    i_s = max(0,min(lvp-1,int(i_v*div)))
    arsp_vet[i_v]=song_points[i_s][2]
"""

"""
    print("Running dtw...")

    xlist = []
    for sp in song_points:
        xlist.append(sp)
    ylist = []
    for vp in video_points:
        ylist.append(vp)

    manhattan_distance = lambda x, y: calc_similarity_sv(x,y)

    d, cost_matrix, acc_cost_matrix, path = dtw(xlist, ylist, dist=manhattan_distance)
            
    song_points_f = []
    video_points_f = []

    for i in range(len(path[0])-1):
        if (path[0][i] != path[0][i+1]):
            song_points_f.append(song_points[path[0][i]])
            video_points_f.append(video_points[path[1][i]])

    return song_points_f, video_points_f
"""

"""
def generate_plots(i,song_points,video_points,ftit1,plots_dir):

    i1,i2,k,n = i[0],i[1],i[2],i[3]

    out_imgs = []

    print("  Init part "+str(k)+"/"+str(n))

    for i in range(i1,i2):

        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(ftit1, fontsize=16)

        grid = plt.GridSpec(8,1, wspace=0.2, hspace=0.1)

        ax1 = fig.add_subplot(grid[0:6,0])
        ax2 = fig.add_subplot(grid[6,0])
        ax3 = fig.add_subplot(grid[7,0])

        s_vals = []
        s_arss = []
        v_vals = []
        v_arss = []
        v_imgs = []

        for j in range(min(len(song_points),len(video_points))):
            song_point = song_points[j]
            video_point = video_points[j]
            
            s_vals.append(song_point[1])
            s_arss.append(song_point[2])
            v_vals.append(video_point[1])
            v_arss.append(video_point[2])
            v_imgs.append(video_point[5])

        limtmp1 = 1.2

        imb1 = plt.imread(v_imgs[i])
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.imshow(imb1)

        ax2.set_ylabel("Valence")
        ax2.set_ylim([-limtmp1, +limtmp1])
        ax2.plot(s_vals,label="audio",color="green")
        ax2.plot(v_vals,label="video",color="blue")
        ax2.legend(loc='lower right') 
        ax2.axvline(i, c='m')

        ax3.set_xlabel("Frame id")
        ax3.set_ylabel("Arousal")
        ax3.set_ylim([-limtmp1, +limtmp1])
        ax3.plot(s_arss,label="audio",color="green")
        ax3.plot(v_arss,label="video",color="blue")
        ax3.legend(loc='lower right') 
        ax3.axvline(i, c='m')

        img_fnam = plots_dir+str(i)+".png"
        
        plt.savefig(img_fnam)

        out_imgs.append(img_fnam)

    print("  Done part "+str(k)+"/"+str(n))

    return out_imgs
"""

"""
def evaluate_pair(song_points_f,video_points_f,vtit1,stit1,mtit1):

    fig = plt.figure(figsize=(16, 8))
    fig.suptitle(vtit1+"-"+stit1+"-"+mtit1, fontsize=16)

    ax1 = fig.add_subplot(3,1,1)
    ax2 = fig.add_subplot(3,1,2)
    ax3 = fig.add_subplot(3,1,3)

    s_vals = []
    s_arss = []
    v_vals = []
    v_arss = []
    
    p_sims = []

    for i in range(min(len(song_points_f),len(video_points_f))):
        song_point = song_points_f[i]
        video_point = video_points_f[i]
        
        s_vals.append(song_point[1])
        s_arss.append(song_point[2])
        v_vals.append(video_point[1])
        v_arss.append(video_point[2])

        p_sims.append(combiner.optimizer.calc_similarity_sv(song_point,video_point))

    m_sim = round(np.mean(p_sims),global_round_numbs)

    limtmp1 = 1.2

    ax1.set_ylabel("Valence")
    ax1.set_ylim([-limtmp1, +limtmp1])
    ax1.plot(s_vals,label="audio",color="green")
    ax1.plot(v_vals,label="video",color="blue")
    ax1.legend(loc='lower right')

    ax2.set_ylabel("Arousal")
    ax2.set_ylim([-limtmp1, +limtmp1])
    ax2.plot(s_arss,label="audio",color="green")
    ax2.plot(v_arss,label="video",color="blue")
    ax2.legend(loc='lower right')     

    ax3.set_xlabel("Frame id")
    ax3.set_ylabel("Similarity")
    ax3.set_ylim([0,1])
    ax3.plot(p_sims,label="average: "+str(m_sim),color="red")
    ax3.legend(loc='lower right')

    ftit1 = vtit1+"_"+stit1+"_"+mtit1

    print(str("Similarity for "+vtit1+"-"+stit1+"-"+mtit1+": "+str(m_sim)))

    plt.savefig(out_dir+ftit1+".png")
"""

"""
def get_best_sim_p(i,song_list,video_fnam,video_points,sel_method):

    i1,i2,k,n = i[0],i[1],i[2],i[3]

    best_sim=0
    best_i=i1
    for i in range(i1,i2):
        song_fnam = song_list[i]
        profgen_a = combiner.profgen.profgen("audio",64,song_fnam,False,False)
        song_points = profgen_a.get_emotion_profile()
        song_points = preprocess_points(song_points,"s")
        if(validate_pair(song_points,video_points)):
            pair_sim = combine_pair(song_fnam,song_points,video_fnam,video_points,sel_method,False)
            print("Song "+str(i-i1)+"/"+str(i2-i1)+": "+str(song_fnam)+": "+str(pair_sim))
            if(pair_sim>best_sim):
                best_i=i
                best_sim = pair_sim
    
    return (best_sim,best_i)

def select_best_song(video_fnam,video_points,songs_dir,shuff=True,only_cache=False):

    sel_method = "uniform"

    print("Searching for best candidate song, with "+str(sel_method)+" method... ")

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

    parallel=True
    if(parallel==False):
        best_sim=0
        best_i=0
        for i in range(len(song_list)):
            print("Calculating similarity for song "+str(i+1)+"/"+str(len(song_list)))
            song_fnam = song_list[i]
            profgen_a = combiner.profgen.profgen("audio",64,song_fnam,False,True)
            song_points = profgen_a.get_emotion_profile()
            song_points = preprocess_points(song_points,"s")
            if(validate_pair(song_points,video_points)):
                pair_sim = combine_pair(song_fnam,song_points,video_fnam,video_points,sel_method,False)
                print("  Curr Song "+str(song_fnam)+": "+str(pair_sim))
                if(pair_sim>best_sim):
                    best_i=i
                    best_sim = pair_sim
            
            best_song = song_list[best_i]
            print("  Best Song "+str(best_song)+": "+str(best_sim))
    else:
        njobs = num_cores
        ljobs = utils.get_ljobs(0,len(song_list),njobs)

        louts = Parallel(n_jobs=njobs)(
                delayed(get_best_sim_p)(i,song_list,video_fnam,video_points,sel_method)
                    for i in ljobs)

        best_song = louts[0]
        best_sim = louts[0][0]
        best_i = louts[0][1]
        for lout in louts:
            pair_sim = lout[0]
            pair_i = lout[1]
            if(pair_sim>best_sim):
                best_sim = pair_sim
                best_i = pair_i

        best_song = song_list[best_i]
        print("  Best Song "+str(best_song)+": "+str(best_sim))

    return best_song
"""

"""
Vídeos definitivos
    ~12 Vídeos
    
Músicas definitivas
    ~4K Músicas

Métricas Quantitativas
    Qualidade da classificação das emoções (poderia também ser reportado como resultados quantitativos)
        Acurácias de teste obtidas com a rede que classifica o áudio
        Acurácias de teste obtidas com a rede que classifica o vídeo
    
    Para cada um dos N vídeos escolhidos:
        Qualidade do casamento
            Similaridade no plano VA - comparação estatística com IC da similaridade média
                Essa seria a principal, para cada vídeo+conjunto de músicas:
                    Calcular a similaridade no plano valence-aroual usando distância euclidiana para cada frame vídeo/música,
                    depois calcular e reportar o intervalo de confiança da similaridade média.
            Tamanho do vídeo final em relação à música
                Avaliar se o tamanho da música coincidiu com o tamanho do vídeo,
                calcular e reportar o erro (tamanho do corte que houve na música ou no vídeo)
        Qualidade visual do vídeo
            Suavidade das transições entre os frames ao longo do tempo
                Calcular, para cada transição, a similaridade do frame atual com o frame seguinte do vídeo,
                depois calcular e reportar o intervalo de confiança da similaridade média.
            Estabilidade da câmera ao longo do tempo
                Calcular, para cada frame, o nível de estabilidade da câmera
                e reportar de forma similar ao item anterior
        Mostrar os resultados em forma de tabela (com 3 dimensões), onde cada linha é um vídeo, cada coluna uma das métricas
        em cada célula linha/coluna mostrar os resultados de cada um dos métodos (que são 4 no total)

Métricas Qualitativas (Para isso teria que implementar o site)
    Para um conjunto de N trechos de vídeo de duração L segundos, gerados usando cada um dos métodos:
        Exibir o trecho de vídeo para o usuário e solicitar as seguintes avaliações:
            Qualidade do casamento:
                Pedir ao usuário para atribuir uma nota de 0 a 10 que defina o quanto ele considera que a música esteja adequada ao vídeo
                Nesse caso pode ser interessante incluir a busca pela melhor música e comparar também com uma escolha aleatória
            Qualidade visual do vídeo:
                Pedir ao usuário para atribuir uma nota de 0 a 10 que defina o quanto ele considera que o vídeo está visualmente agradável de se assistir
    
    Obs.: O usuário não saberá qual método foi usado para gerar cada vídeo

    Mostrar os resultados também em forma de tabela, de forma similar aos quantitativos

"""

"""
def optimize_acccost(self,song_points,video_points):

        print("Note: Using optimize_acccost (not considering emotion similarity)")

        n_sngfms = len(song_points)
        n_vidfms = len(video_points)
        n_vsdiff = n_vidfms/n_sngfms

        acc_vet = []
        for i in range(len(song_points)):
            song_point = song_points[i]
            acc = song_point[2]
            acc = (acc+1)/2
            acc = int(9*acc)+1
            acc_vet.append(acc)

        video_points_f = []
        song_points_f = song_points

        acc_scl=1

        i_s=0
        i_v=0
        while(i_s<len(song_points) and i_v<len(video_points)):
            utils.show_progress("Running greedy optimizer... ",i_s,len(song_points))
            video_points_f.append(video_points[int(i_v)])
            i_v+=acc_vet[i_s]*acc_scl
            i_s+=1

        return song_points_f, video_points_f 
    """

    """
    def calc_revrate(video_points):
    print("Calculating reversion rate...")
    rev_fms=0
    for i in range(len(video_points)):
        test = False
        for j in range(i,len(video_points)):
            frmnmb_i = int(utils.path2fnam(video_points[i][5])[:-4])
            frmnmb_j = int(utils.path2fnam(video_points[j][5])[:-4])
            if(frmnmb_j<frmnmb_i):
                test = True
                break
        if(test==True):
            rev_fms+=1
    tot_fms = len(video_points)
    rev_rat = round(rev_fms/tot_fms,global_round_numbs)
    return rev_rat

def calc_reprate(video_points):
    print("Calculating repetition rate...")
    rep_fms=0
    for i in range(len(video_points)):
        test=False
        for j in range(len(video_points)):
            if(video_points[i][5]==video_points[j][5] and j!=i):
                test=True
                break
        if(test==True):
            rep_fms+=1
    tot_fms = len(video_points)
    rep_rat = round(rep_fms/tot_fms,global_round_numbs)
    return rep_rat
"""

"""
def calc_sngcut(song_points,song_fnam):
    print("Calculating song loss...")
    import librosa
    y, sr = librosa.load(song_fnam)
    len_orig = librosa.get_duration(y,sr)
    len_res = len(song_points)/global_out_fps
    diff = round(max(0,len_orig-len_res),global_round_numbs)
    return diff

def calc_vidcut(video_points,video_fnam):
    print("Calculating video loss...")
    vid_cap = cv2.VideoCapture(video_fnam)
    lfm_orig = int(cv2.VideoCapture.get(vid_cap,int(cv2.CAP_PROP_FRAME_COUNT)))-1
    lfm_res = video_points[-1][5]
    lfm_res = int(utils.path2fnam(lfm_res)[:-4])
    diff = round((max(0,lfm_orig-lfm_res))/global_out_fps,global_round_numbs)
    return diff
"""

"""
if(opt_method=="ours"):
    print("Saving points...")
    (pd.DataFrame(song_points_f )).to_csv("out/4/points_s.csv", index=None, header=False)
    (pd.DataFrame(video_points_f)).to_csv("out/4/points_v.csv", index=None, header=False)
"""