from header import *

def preprocess_input_image(i_input):
    i_input = Image.open(i_input).convert('RGB')
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]
    input_size = 224
    img_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)])
    
    i_input = img_transform(i_input)

    return i_input

def preprocess_input_audio(song_filepath):

    num_features = 48

    i=-1
    while(song_filepath[i]!="-"):
        i-=1
    time_id = int(song_filepath[i+1:])
    song_filename = song_filepath[:i]
    j=i
    while(song_filepath[j]!="/"):
        j-=1
    song_id=song_filepath[j+1:i-4]
    song_dir = song_filepath[:j+1]
    filepart_dir = song_dir+song_id+"/"
    for i in range(5):
        try:
            if not(os.path.exists(filepart_dir)):
                os.makedirs(filepart_dir)
            break
        except:
            pass

    filepart_wav = filepart_dir+"frames/"+str(time_id)+".wav"
    filepart_csv = filepart_dir+"frames/"+str(time_id)+".csv"

    for i in range(5):
        try:
            if(os.path.exists(filepart_csv)):
                df = pd.read_csv(filepart_csv)
                filtered_features = df.iloc[:,0]
                filtered_features = torch.FloatTensor(filtered_features)
                return filtered_features[:num_features]
            break
        except:
            pass

    try:
        import librosa
        import essentia.standard

        step_length = 6.0
        offset = float(time_id/1000) - step_length 
        duration = step_length
        y, sr = librosa.load(song_filename,offset=offset,duration=duration)
        
        if not(os.path.exists(filepart_wav)):
            librosa.output.write_wav(filepart_wav,y,sr)

        loader = essentia.standard.MonoLoader(filename=filepart_wav)
        audio = loader()
        features, features_frames = essentia.standard.MusicExtractor(
            lowlevelStats=["mean", "stdev","min", "max"],
            rhythmStats=["mean", "stdev", "min", "max"],
            tonalStats=["mean", "stdev", "min", "max"])(filepart_wav)     

        filtered_features = []
        deleted_features = [] 
        allowed_features = []

        vtmp1 = [
            "erbbands",
            "melbands",
            "mfcc",
            "onset_rate",
            "beats_loudness",
            "_band_ratio",
            "key_edma.strength",
            "key_krumhansl.strength",
            "key_temperley.strength",
            "chords_histogram"
            ]

        allowed_features = []
        for j in features.descriptorNames():
            if isinstance(j, (int, float, bool, str)):
                for k in vtmp1:
                    if k in j:
                        allowed_features.append(j)    

        for j in features.descriptorNames():
            if isinstance(features[j], (int, float, bool, str)) and j in allowed_features:
                filtered_features.append(features[j])
            else:
                deleted_features.append(j)
    except:
        print("  Note: Returning zeros feature vector!")
        filtered_features = [0]*num_features

    (pd.DataFrame(filtered_features)).to_csv(filepart_csv,index=None,header=True)

    filtered_features = torch.FloatTensor(filtered_features)

    return filtered_features[:num_features]

class c(Dataset):

    def __init__(self,data_list,data_type):
        self.data_list = data_list
        self.data_len = len(data_list)
        self.input_size = None
        if(data_type=="video"):
            self.preprocess_input = preprocess_input_image
        elif(data_type=="audio"):
            self.preprocess_input = preprocess_input_audio
            
    def __getitem__(self, index):
        i_input = self.data_list[index][0]
        i_label = self.data_list[index][1]
        i_input = self.preprocess_input(i_input)
        return (i_input,i_label)
    
    def __len__(self):
        return self.data_len