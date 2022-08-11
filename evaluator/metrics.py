from header import *
from tqdm import tqdm


def calc_emosim(song_points, video_points):
    p_sims = []
    for i in tqdm(range(min(len(song_points), len(video_points))), desc='Calculating emotion similarity...'):
        p_sim = combiner.optimizer.calc_similarity_sv(song_points[i], video_points[i])
        p_sims.append(p_sim)
    m_sim = round(np.mean(p_sims), global_round_numbs)
    print("EmoSim: {}".format(m_sim))
    return m_sim


def calc_sngtrc(song_points, len_sp_i):
    print("Calculating song truncation...")
    sngtrc = max(0, len_sp_i-len(song_points))/global_out_fps
    return sngtrc


def calc_vidtrc(video_points, len_vp_i):
    print("Calculating video truncation...")
    lfm_i = len_vp_i-1
    lfm_f = int(utils.path2fnam(video_points[-1][5])[:-4])
    vidtrc = max(0, lfm_i-lfm_f)/global_out_fps
    return vidtrc


def calc_desspd(len_sp_i, len_vp_i):
    print("Calculating desired speedup...")
    desspd = len_vp_i/len_sp_i
    return desspd


def calc_obtspd(video_points, len_vp_i):
    print("Calculating obtained speedup...")
    vidtrc = calc_vidtrc(video_points, len_vp_i)
    vidtrc_p = vidtrc*global_out_fps
    obtspd = len_vp_i/(len(video_points)+vidtrc_p)
    return obtspd


def calc_vidsmt(video_points):

    print("Calculating video smoothness...")

    def calc_sims_p(i, video_points):

        i1, i2, k, n = i[0], i[1], i[2], i[3]

        print("  Init part "+str(k)+"/"+str(n))

        p_sims = []

        for i in range(i1, i2):
            fnam1 = video_points[i][5]
            fnam2 = video_points[i+1][5]
            # if(fnam1!=fnam2):
            if(True):
                img1 = cv2.resize(cv2.imread(fnam1), global_cmp_size)
                img2 = cv2.resize(cv2.imread(fnam2), global_cmp_size)
                p_sim = combiner.optimizer.calc_similarity_vv(img1, img2)
                p_sims.append(p_sim)

        print("  Done part "+str(k)+"/"+str(n))

        return p_sims

    njobs = num_cores
    ljobs = utils.get_ljobs(0, len(video_points)-1, njobs)
    louts = Parallel(n_jobs=njobs)(
        delayed(calc_sims_p)(i, video_points)
        for i in ljobs)
    p_sims = []
    for lout in louts:
        for p_sim in lout:
            p_sims.append(p_sim)
    m_sim = round(np.mean(p_sims), global_round_numbs)

    print('Smoothness: {}'.format(m_sim))
    return m_sim


def calc_vidshk(video_points):

    print("Calculating video shaking...")

    import evaluator.thirdpart
    from evaluator.thirdpart import shaking
    frame_list = []
    for video_point in video_points:
        frame_list.append(video_point[5])

    shak = round(shaking.calc_video_shaking(frame_list), global_round_numbs)

    print('Shaking Ratio: {}'.format(shak))
    return shak


def calc_vidfid(video_points, video_fnam):

    print("Calculating video fid_score...")

    import evaluator.thirdpart
    from evaluator.thirdpart import fidscore

    orig_video_frames_dir = cache_dir+"video/"+utils.path2fnam(video_fnam)[:-4]+"/frames/"

    files1 = []
    files2 = []

    for root, dirs, files in os.walk(orig_video_frames_dir):
        for file in files:
            if file.endswith(".jpg"):
                files1.append(os.path.join(root, file))

    for video_point in video_points:
        files2.append(video_point[5])

    fdsc = round(fidscore.calculate_fid_for_files(files1, files2), global_round_numbs)

    print('FID: {}'.format(fdsc))
    return fdsc


def calc_vidfid2(video_points1, video_points2):

    print("Calculating video fid_score...")

    import evaluator.thirdpart
    from evaluator.thirdpart import fidscore

    files1 = []
    files2 = []

    for video_point in video_points1[:1000]:
        files1.append(video_point[5])

    for video_point in video_points2[:1000]:
        files2.append(video_point[5])

    fdsc = round(fidscore.calculate_fid_for_files(files1, files2), global_round_numbs)

    return fdsc
