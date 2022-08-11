from header import *
import dijkstra
import scipy
from skimage import measure
import combiner.thirdpart
from combiner.thirdpart import dtw
from tqdm import tqdm


def calc_optim_w(song_len, video_len):
    w_max = global_optimizer_w
    w = min(w_max, max(4, int(2*video_len/song_len)))
    return w


def euclidean_distance(p1, p2):
    dist = math.sqrt(((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2))
    return dist


def calc_similarity_sv(song_point, video_point, mode="cont"):

    s_v = song_point[1]*0.5
    s_a = song_point[2]*0.5

    if(mode == "cont"):
        v_v = video_point[1]
        v_a = video_point[2]
    elif(mode == "disc"):
        v_v = video_point[3]
        v_a = video_point[4]

    max_dist = 2.828427  # Distance from [-1,-1] to [+1.+1]
    dist = euclidean_distance([s_v, s_a], [v_v, v_a])
    dist = dist/max_dist
    psim = 1-dist

    return psim


def calc_similarity_vv(img1, img2):
    p_sim = measure.compare_ssim(img1, img2, multichannel=True)
    return p_sim


class optimizer_uniform():

    def optimize(self, song_points, video_points):

        n_sngfms = len(song_points)
        n_vidfms = len(video_points)
        n_nxtfms = n_vidfms/n_sngfms

        song_points_f = []
        video_points_f = []

        i_frame_s = 0
        i_frame_v = 0
        while(i_frame_s < n_sngfms and i_frame_v < n_vidfms):
            utils.show_progress("Running uniform optimizer... ", i_frame_s, n_sngfms)
            song_points_f.append(song_points[int(i_frame_s)])
            video_points_f.append(video_points[int(i_frame_v)])
            i_frame_s += 1
            i_frame_v += n_nxtfms

        return song_points_f, video_points_f


class optimizer_uniform_p():

    def optimize(self, song_points, video_points):

        n_sngfms = len(song_points)
        n_vidfms = len(video_points)
        n_drpfms = n_vidfms-n_sngfms

        n_nxtfms = n_vidfms/n_sngfms

        video_points_f = []
        song_points_f = []

        i_frame_s = 0
        i_frame_v = 0
        while(i_frame_s < n_sngfms):
            utils.show_progress("Running uniform plus optimizer... ", i_frame_s, n_sngfms)
            max_sim = 0
            j_frame_best = i_frame_v
            for j_frame_v in range(int(i_frame_v), min(int(i_frame_v+n_nxtfms), n_vidfms-1)):
                pair_sim = calc_similarity_sv(song_points[i_frame_s], video_points[int(j_frame_v)])
                if(pair_sim > max_sim):
                    max_sim = pair_sim
                    j_frame_best = j_frame_v

            video_points_f.append(video_points[int(j_frame_best)])
            song_points_f.append(song_points[i_frame_s])

            i_frame_s += 1
            i_frame_v += n_nxtfms

            if(i_frame_s > n_sngfms):
                break

        return song_points_f, video_points_f


class optimizer_greedy():

    def optimize_acccost(self, song_points, video_points):

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

        acc_scl = 1

        i_s = 0
        i_v = 0
        while(i_s < len(song_points) and i_v < len(video_points)):
            utils.show_progress("Running greedy optimizer... ", i_s, len(song_points))
            video_points_f.append(video_points[int(i_v)])
            i_v += acc_vet[i_s]*acc_scl
            i_s += 1

        return song_points_f, video_points_f

    def optimize_simcost(self, song_points, video_points):

        n_sngfms = len(song_points)
        n_vidfms = len(video_points)
        n_vsdiff = n_vidfms/n_sngfms

        song_points_f = []
        video_points_f = []

        i_s = 0
        i_v = 0
        w = calc_optim_w(n_sngfms, n_vidfms)
        while(i_s < len(song_points) and i_v < len(video_points)):
            utils.show_progress("Running greedy optimizer... ", i_s, len(song_points))
            maxsim = calc_similarity_sv(song_points[i_s], video_points[i_v])
            i_maxsim = i_v
            for i_v2 in range(i_v, min(i_v+w, len(video_points))):
                pairsim = calc_similarity_sv(song_points[i_s], video_points[i_v2])
                if(pairsim > maxsim):
                    maxsim = pairsim
                    i_maxsim = i_v2
            song_points_f.append(song_points[i_s])
            video_points_f.append(video_points[i_maxsim])
            i_v = i_maxsim+1
            i_s += 1

        return song_points_f, video_points_f

    def optimize(self, song_points, video_points):
        return self.optimize_simcost(song_points, video_points)


class optimizer_dijkstra():

    def diag_bold_matrix(self, size, radius):
        matrix = np.zeros(size)
        matrix[0][0] = 1
        matrix[matrix.shape[0]-1][matrix.shape[1]-1] = 1
        for i in range(1, matrix.shape[0]-1):
            for j in range(radius):
                a = min(size[1]-1, math.floor(i*size[1]/size[0])+j)
                matrix[i][a] = 1
        return matrix

    def mask_bold(self, matrix, radius):
        boldmatrix = self.diag_bold_matrix(matrix.shape, radius)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                matrix[i][j] = matrix[i][j]*boldmatrix[i][j]
        return matrix

    def get_nodes(self, matrix):
        nodes = []
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i][j] != 0:
                    nodes.append((i, j))
        return nodes

    def search_next_line_nodes(self, node, matrix):
        next_line_index = min(node[0]+1, matrix.shape[0]-1)
        search_area = []
        for i in range(matrix.shape[1]):
            search_area.append((next_line_index, i))
        return search_area

    def valid_edge(self, edge, nodes, maxjump):
        if edge[1] in nodes and edge[0][1] < edge[1][1] and abs(edge[0][1] - edge[1][1]) <= maxjump:
            return True

    def connect_nodes(self, nodes, matrix, maxjump):
        edges = []
        cont = 0
        for node1 in nodes:
            utils.show_progress("Connecting nodes... ", cont, len(nodes))
            cont = cont + 1
            next_line_nodes = self.search_next_line_nodes(node1, matrix)
            for node2 in next_line_nodes:
                if self.valid_edge((node1, node2), nodes, maxjump):
                    edges.append((node1, node2))
        return edges

    def separate(self, edges):
        x = []
        y = []
        for i in edges:
            x.append(i[0])
            y.append(i[1])
        return y, x

    def dijkstra_calc(self, edges, S):
        graph = dijkstra.Graph()
        for i in edges:
            cost = 1 - S[i[1][0]][i[1][1]]
            graph.add_edge(i[0], i[1], cost)
        dijkstra_result = dijkstra.DijkstraSPF(graph, (0, 0))
        return dijkstra_result.get_path(edges[-1][1])

    def dijkstra_combiner(self, S, radius, maxjump):

        S = self.mask_bold(S, radius)
        nodes = self.get_nodes(S)
        edges = self.connect_nodes(nodes, S, maxjump)
        path = self.dijkstra_calc(edges, S)
        v, m = self.separate(path)

        print(v)

        return m, v

    def compute_similarity(self, song_points, video_points):

        sim_mat = np.zeros((len(song_points), len(video_points)))

        for i in range(len(song_points)):
            utils.show_progress("Computing Similarities... ", i, len(song_points))
            for j in range(len(video_points)):
                sim_mat[i][j] = calc_similarity_sv(song_points[i], video_points[j])

        return sim_mat

    def optimize(self, song_points, video_points):

        print("Running dijkstra optimizer...")

        sim_mat = self.compute_similarity(song_points, video_points)

        w = calc_optim_w(len(song_points), len(video_points))

        song_ids, video_ids = self.dijkstra_combiner(sim_mat, radius=int(w/2), maxjump=100000)

        print("Done")

        song_points_f = []
        video_points_f = []

        for s_id in song_ids:
            song_points_f.append(song_points[s_id])

        for v_id in video_ids:
            video_points_f.append(video_points[v_id])

        return song_points_f, video_points_f


class optimizer_dijkstra_c():

    def optimize(self, song_points, video_points, resize_dim=(128, 64)):

        print("Running dijkstra approximated optimizer...")

        d = optimizer_dijkstra()
        sim_mat = d.compute_similarity(song_points, video_points)

        print("Shrinking the similarity matrix...")
        sim_mat = cv2.resize(sim_mat, resize_dim)

        w = calc_optim_w(len(song_points), len(video_points))

        song_ids, video_ids = d.dijkstra_combiner(sim_mat, radius=int(w), maxjump=8)

        print("Resizing the results...")

        video_ids = [int(float(x) * (len(song_points)/resize_dim[1])) for x in video_ids]
        video_ids = np.resize(video_ids, len(song_points))

        song_ids = np.array(range(len(song_points)))
        song_ids = [int(float(x)) for x in song_ids]

        print("Looking for mistakes...")

        for i in range(len(song_ids)-1):
            if song_ids[i] == song_ids[i+1]:
                print("Found a mistake in i = ", i)

        print("Done")

        song_points_f = []
        video_points_f = []

        for s_id in song_ids:
            song_points_f.append(song_points[s_id])
        for v_id in video_ids:
            video_points_f.append(video_points[v_id])

        return song_points_f, video_points_f


class optimizer_dtw():

    def optimize(self, song_points, video_points):

        print("Running dtw optimizer...")

        song_points.reverse()
        video_points.reverse()

        def manhattan_distance(x, y): return 1-calc_similarity_sv(x, y)

        w = calc_optim_w(len(song_points), len(video_points))

        d, cost_matrix, acc_cost_matrix, path = dtw.dtw(song_points, video_points, w2=w, dist=manhattan_distance)

        song_points_f = []
        video_points_f = []

        for i in range(len(path[0])-1):
            if (path[0][i] != path[0][i+1]):
                song_points_f.append(song_points[path[0][i]])
                video_points_f.append(video_points[path[1][i]])

        song_points_f.reverse()
        video_points_f.reverse()

        return song_points_f, video_points_f


class sparse_matrix():

    def __init__(self, size, defval):
        self.i_list = []
        self.i_exst = [0]*size
        self.defval = defval

    def set_val(self, i, j, val):
        if(self.i_exst[i] == 0):
            self.i_list.append([i, [[j, val]]])
            self.i_exst[i] = 1
        else:
            for item_i in self.i_list:
                if(item_i[0] == i):
                    for item_j in item_i[1]:
                        if(item_j[0] == j):
                            item_j[1] = val
                            return
                    item_i[1].append([j, val])
                    return

    def get_val(self, i, j):
        if(self.i_exst[i] == 0):
            return self.defval
        else:
            for item_i in self.i_list:
                if(item_i[0] == i):
                    for item_j in item_i[1]:
                        if(item_j[0] == j):
                            return item_j[1]
                    return self.defval

    def set_val_v(self, i, v):
        if(self.i_exst[i] == 0):
            j_list = v
            self.i_list.append([i, j_list])
            self.i_exst[i] = 1
        else:
            for item_i in self.i_list:
                if(item_i[0] == i):
                    for item_j in v:
                        item_i[1].append(item_j)
                    return

    def get_min(self, i):

        min_h = float('inf')
        argmin_h = i-1

        for item_i in self.i_list:
            if(item_i[0] >= 0 or item_i[0] <= i-1):
                for item_j in item_i[1]:
                    if(item_j[0] == i):
                        if(item_j[1] < min_h):
                            min_h = item_j[1]
                            argmin_h = item_i[0]

        return min_h, argmin_h


class optimizer_dinprog():

    def __init__(self):

        self.p_totspd = .2  # 0.01    #Weight for cost of total speedup
        self.p_frmsim = .2  # 0.01    #Weight for cost of inter-frame similarity
        self.p_relspd = 0.00  # Weight for cost of relative speedup (not used)
        self.p_emosim = .6  # 1.00    #Weight for cost of emotion similarity

    def normalize_min_max(self, matrix):
        return (matrix - matrix.min()) / (matrix.max() - matrix.min())

    def compute_vsim_p(self, i, F, w, vi):
        i1, i2, k, n = i[0], i[1], i[2], i[3]
        Mat = np.zeros((i2-i1, F))
        #print("  Init part "+str(k)+"/"+str(n))
        for i in range(i1, i2):
            for j in range(max(i-w, 0), min(i+w, F)):
                Mat[i-i1, j] = calc_similarity_vv(vi[i], vi[j])
        #print("  Done part "+str(k)+"/"+str(n))
        return Mat

    def create_inter_frame_similarity_matrix(self, video_images, F, w, batch_size):
        # Requires TensorFlow (==2.5?)

        S_i = np.eye(F, dtype=np.float32)  # Creates an identity matrix

        max_skip_range = w*2
        num_frames_per_batch = batch_size//(max_skip_range)
        num_batches = math.ceil((F-max_skip_range)/num_frames_per_batch)

        # session = tf.Session()
        with sess.as_default():
            # Compute for all batches but one (the last one)
            for i in tqdm(range(num_batches-1), desc='Computing SSIM (batchwise)'):
                idx_start_i = i*num_frames_per_batch
                idx_end_i = min((i+1)*num_frames_per_batch, (F-max_skip_range))
                # im1 = tf.repeat(tf.convert_to_tensor(video_images[idx_start_i:idx_end_i], dtype=tf.uint8), max_skip_range, axis=0)  # Repeat/ row images "max_skip_range" times each for batch computing
                im1 = tf.convert_to_tensor(np.repeat(video_images[idx_start_i:idx_end_i], max_skip_range, axis=0), dtype=tf.uint8)  # Repeat/ row images "max_skip_range" times each for batch computing

                im2_list = []
                for j in range(idx_start_i, idx_end_i):  # From 2nd of the batch to the number of frames in this batch
                    im2_list.append(tf.convert_to_tensor(video_images[j+1:j+max_skip_range+1], dtype=tf.uint8))

                im2 = tf.concat(im2_list, axis=0)
                ssim = tf.image.ssim(im1, im2, max_val=255).eval()

                for idx_k, k in enumerate(range(idx_start_i, idx_end_i)):
                    S_i[k][k+1:k+max_skip_range+1] = ssim[idx_k*max_skip_range:(idx_k+1)*max_skip_range]

        # Compute for the last batch
        for i in range((num_batches-1)*num_frames_per_batch, F-1):
            idx_start_j = i+1
            idx_end_j = min(i+max_skip_range+1, F)

            im1 = tf.convert_to_tensor(np.repeat(video_images[i:i+1], idx_end_j-idx_start_j, axis=0), dtype = tf.uint8)  # Repeat/ row images "max_skip_range" times each for batch computing

            im2 = tf.convert_to_tensor(video_images[idx_start_j:idx_end_j], dtype=tf.uint8)

            S_i[i][idx_start_j:idx_end_j] = tf.image.ssim(im1, im2, max_val=255)

        return S_i

    def compute_slice_dinprog(self, W_Ci_Cs, W_Ce, Dk_minus_1, w, k, i, F):

        # Find the minimum in the previous slice (k-1)
        min_h = Dk_minus_1[i-1, i]
        argmin_h = i-1
        for h in range(2, min(w, i+1)):
            if Dk_minus_1[i-h, i] < min_h:
                min_h = Dk_minus_1[i-h, i]
                argmin_h = i-h

        Dki = np.zeros(F, dtype=np.float32)
        Tki = np.zeros(F, dtype=np.uint32)

        # Populate current slice
        for j in range(i+1, min(i+1+w, F)):

            c = W_Ci_Cs[i, j] + W_Ce[k, j]

            Dki[j] = c + min_h
            Tki[j] = argmin_h

        return Dki, Tki

    def optimize_cpu(self, song_points, video_points, part_id=1, batch_size=1024):

        c_max = 200
        song_points = song_points
        video_points = video_points

        M = len(song_points)
        F = len(video_points)
        g = 2
        w = calc_optim_w(M, F)

        C_s = np.zeros((F, F), dtype=np.float32)
        C_r = np.zeros((F, F), dtype=np.float32)
        S_i = np.zeros((F, F), dtype=np.float32)
        S_e = np.zeros((M, F), dtype=np.float32)

        if self.p_frmsim:
            print("Computing S_i...")
            video_images = []

            video = cv2.VideoCapture(args.v)
            for i in tqdm(range(F), desc='Loading Video Images'):
                _, frame = video.read()

                video_images.append(cv2.resize(frame, global_cmp_size))#, dtype=tf.uint8)

            video.release()

            video_images = np.array(video_images, dtype=np.uint8)
            S_i = self.create_inter_frame_similarity_matrix(video_images, F, w, batch_size)
            C_i = 1-S_i
            C_i = self.normalize_min_max(C_i)

        if self.p_totspd:
            v = F/M

            for i in tqdm(range(F), desc='Computing C_s...'):
                for j in range(i, min(i+w*2, F)):
                    cs = min(((j-i)-v)**2, c_max)
                    C_s[i, j] = cs

            C_s = self.normalize_min_max(C_s)

        if self.p_relspd:

            for i in tqdm(range(F), desc='Computing C_r...'):
                for j in range(i, min(i+w*2, F)):
                    cr = 1+0.5*(video_points[i][2]+video_points[j][2])
                    C_r[i, j] = cr

            C_r = self.normalize_min_max(C_r)

        if self.p_emosim:

            for k in tqdm(range(M), desc='Computing S_e...'):
                for j in range(k+1, min(i+w*2, F)):
                    S_e[k, j] = calc_similarity_sv(song_points[k], video_points[j])

            C_e = 1-S_e
            C_e = self.normalize_min_max(C_e)

        # Compute weighted costs to avoid repeated computation
        W_Ci_Cs = self.p_frmsim*C_i + self.p_totspd*C_s
        W_Ce = self.p_emosim*C_e

        # Create 3D Dynamic Cost Matrix and the Traceback Matrix
        D = [None for _ in range(M)]
        T = [None for _ in range(M)]

        # Fill the first slice, song frame
        D[0] = scipy.sparse.csr_matrix(C_s)

        # Computing the following slices
        for k in tqdm(range(1, M), desc="Computing Path using {} cores...".format(num_cores)):

            # Parallelize computing at each slice
            Dks_Tks = Parallel(n_jobs=num_cores)(
                delayed(self.compute_slice_dinprog)(W_Ci_Cs, W_Ce, D[k-1], w, k, i, F)
                for i in range(g, F))

            Dk, Tk = zip(*Dks_Tks)

            # Prepend slice with zeros (necessary step :(, sorry)
            Dk = np.concatenate((np.zeros((g, F), dtype=np.float32), np.array(Dk)), axis=0)
            Tk = np.concatenate((np.zeros((g, F), dtype=np.float32), np.array(Tk)), axis=0)

            # Fill the slice
            D[k] = scipy.sparse.csr_matrix(Dk, dtype=np.float32)
            T[k] = scipy.sparse.csr_matrix(Tk, dtype=np.uint32)

        # Freely choose the last selected frame within a range
        s = F-g
        d = s+1
        min_sd = D[M-1][s, d]
        for i in range(F-g, F):
            for j in range(i+1, min(i+1+w, F)):
                if D[M-1][i, j] < min_sd:
                    min_sd = D[M-1][i, j]
                    s = i
                    d = j
        P = [d]
        k = M-1

        # import pdb ; pdb.set_trace()
        while(s > g):
            P.append(s)

            b = T[k-1][s, d]
            d = s
            s = b
            k = k-1

        P.reverse()
        print(P)

        del D
        del T

        video_points_f = []
        song_points_f = []

        while(len(P) < M):
            P.insert(0, 0)

        for item in P:
            video_points_f.append(video_points[item])

        song_points_f = song_points

        return song_points_f, video_points_f

    def optimize(self, song_points, video_points, part_id=1, batch_size=256):

        c_max = 200
        # song_points = song_points[:50] ## for debugging
        # video_points = video_points[:500] ## for debugging

        M = len(song_points)
        F = len(video_points)
        w = calc_optim_w(M, F)
        g = w

        if self.p_frmsim:
            print("Computing S_i...")
            video_images = []

            video = cv2.VideoCapture(args.v)
            for i in tqdm(range(F), desc='Loading Video Images'):
                _, frame = video.read()

                video_images.append(cv2.resize(frame, global_cmp_size))

            video.release()

            video_images = np.array(video_images, dtype=np.uint8)
            S_i = self.create_inter_frame_similarity_matrix(video_images, F, w, batch_size)

            C_i = 1-S_i
            C_i = self.normalize_min_max(C_i)
            del video_images

        if self.p_totspd:
            C_s = np.zeros((F, F), dtype=np.float32)

            v = F/M

            for i in tqdm(range(F), desc='Computing C_s...'):
                for j in range(i, min(i+w*2, F)):
                    cs = min(((j-i)-v)**2, c_max)
                    C_s[i, j] = cs

            C_s = self.normalize_min_max(C_s)
            C_s = np.triu(C_s, k=1)  # Remove unnecessary values

        if self.p_relspd:
            C_r = np.zeros((F, F), dtype=np.float32)

            for i in tqdm(range(F), desc='Computing C_r...'):
                for j in range(i, min(i+w*2, F)):
                    cr = 1+0.5*(video_points[i][2]+video_points[j][2])
                    C_r[i, j] = cr

            C_r = self.normalize_min_max(C_r)
            C_r = np.triu(C_r, k=1)  # Remove unnecessary values

        if self.p_emosim:

            # WARNING: Mode is continuous by default mode="cont"
            song_points_torch = torch.from_numpy(np.array(song_points)[:, 1:3].astype(np.float32)*0.5).to(device)
            video_points_torch = torch.from_numpy(np.array(video_points)[:, 1:3].astype(np.float32)).to(device)
            max_dist = 2.828427

            # Calculate Euclidean Distance (much more quicker than using for loops)
            C_e = torch.cdist(song_points_torch.unsqueeze(0), video_points_torch.unsqueeze(0))[0]
            C_e /= max_dist
            C_e = self.normalize_min_max(C_e)
            C_e = torch.triu(C_e, diagonal=1)  # Remove unnecessary values

        # Compute weighted costs to avoid repeated computation
        W_Ci_Cs = torch.tensor(self.p_frmsim*C_i + self.p_totspd*C_s, dtype=torch.float32, device=device)
        W_Ce = torch.tensor(self.p_emosim*C_e, dtype=torch.float32, device=device)

        # Create 3D Dynamic Cost Matrix and the Traceback Matrix
        D = [None for _ in range(M)]
        T = [None for _ in range(M)]

        # Fill the first slice, song frame
        D[0] = torch.tensor(C_s, device=device)
        T[0] = torch.zeros((F, F), dtype=torch.int32).to_sparse()

        del C_s
        del C_e
        del S_i
        del C_i

        # Computing the following slices
        for k in tqdm(range(1, M), desc="Computing Optimal Path using GPU...".format(num_cores)):

            ThreeD_Cost_Matrix_k = W_Ci_Cs + W_Ce[k].repeat((F, 1))
            ThreeD_Cost_Matrix_k = torch.triu(ThreeD_Cost_Matrix_k, diagonal=1)  # Remove unnecessary values

            D[k] = torch.zeros((F, F), dtype=torch.float32, device=device)
            T[k] = torch.zeros((F, F), dtype=torch.int16, device=device)

            for i in range(g, F-1):
                start_idx = i+1
                end_idx = min(i+w*2+1, F)
                mins_h, argmins_h = torch.min(D[k-1][i-w:i-1, start_idx:end_idx], axis=0)
                D[k][i, start_idx:end_idx] = mins_h + ThreeD_Cost_Matrix_k[i, start_idx:end_idx]
                T[k][i, start_idx:end_idx] = i - w + argmins_h

            # Fill the slice
            T[k] = T[k].to_sparse().cpu()
            D[k-1] = D[k-1].to_sparse().cpu()

        # Freely choose the last selected frame within a range
        s = F-g
        d = s+1
        min_sd = D[M-1][s, d]
        for i in range(F-g, F):
            for j in range(i+1, min(i+1+w, F)):
                if D[M-1][i, j] < min_sd:
                    min_sd = D[M-1][i, j]
                    s = i
                    d = j

        D[M-1] = D[M-1].to_sparse().cpu()

        P = [d]
        k = M-1

        while(s > g):
            P.append(s)

            b = T[k-1][s, d].item()
            d = s
            s = b
            k = k-1

        P.reverse()
        print(P)

        del D
        del T

        video_points_f = []
        song_points_f = []

        while(len(P) < M):
            P.insert(0, 0)

        for item in P:
            video_points_f.append(video_points[item])

        song_points_f = song_points

        return song_points_f, video_points_f


def get_opt_class_dict():

    opt_class_dict = {
        # "uniform": optimizer_uniform,
        # "uniform_p": optimizer_uniform_p,
        # "greedy": optimizer_greedy,
        # "dijkstra": optimizer_dijkstra,
        # "dijkstra_c":optimizer_dijkstra_c,
        # "dtw": optimizer_dtw,
        "ours": optimizer_dinprog,
    }

    return opt_class_dict


def optimize(optimizer, song_points, video_points):

    parallel = False  # True
    if(parallel == True):
        njobs = min(num_cores, max(2, int(len(song_points)/global_par_split_ref)))
        sp_size = int(len(song_points)/njobs)
        vp_size = int(len(video_points)/njobs)

        parts = []

        for i in range(njobs):
            i1 = (i+0)*sp_size
            i2 = (i+1)*sp_size
            if(i == njobs-1):
                i2 = len(song_points)
            sp_part = song_points[i1:i2]
            i1 = (i+0)*vp_size
            i2 = (i+1)*vp_size
            if(i == njobs-1):
                i2 = len(video_points)
            vp_part = video_points[i1:i2]
            parts.append([sp_part, vp_part, i])

        louts = Parallel(n_jobs=njobs)(
            delayed(optimizer.optimize)(part[0], part[1])
            for part in parts)

        song_points_f = []
        video_points_f = []
        for lout in louts:
            for sp in lout[0]:
                song_points_f.append(sp)
            for vp in lout[1]:
                video_points_f.append(vp)
    else:
        song_points_f, video_points_f = optimizer.optimize(song_points, video_points)

    return song_points_f, video_points_f


def pre_accel_video(song_points, video_points):

    song_points_f = []
    video_points_f = []

    nfvi = len(video_points)
    nfvf = min(18000, len(video_points)*0.67)
    spd = nfvi/nfvf

    i = 0
    while(i < nfvi):
        video_points_f.append(video_points[int(i)])
        i += spd

    nfsi = len(song_points)
    nfsf = min(int(nfvf/3), len(song_points))
    song_points_f = song_points[:nfsf]

    print("Song/video lengths (pre-accel): ", len(song_points_f), len(video_points_f))

    return song_points_f, video_points_f


def run(song_points, video_points, opt_method):

    if(global_quick_test == True):
        song_points = song_points[:100]
        video_points = video_points[:500]

    if(global_preaccel_video == True):
        song_points, video_points = pre_accel_video(song_points, video_points)

    if(opt_method == "ours" and len(video_points) > 30000):
        video_points2 = []
        for i in range(0, len(video_points), 2):
            video_points2.append(video_points[i])
        video_points = video_points2

    opt_class_dict = get_opt_class_dict()
    optimizer1 = opt_class_dict[opt_method]()
    song_points_f, video_points_f = optimize(optimizer1, song_points, video_points)

    vidsng_diff = len(video_points_f)-len(song_points_f)
    if(vidsng_diff == 0):
        print(" >> Video and song sizes already match!")
    elif(vidsng_diff > 0):
        print(" >> Video bigger than song, discarding part of video at end")
        video_points_f = video_points_f[0:int(len(song_points_f))]
    else:
        print(" >> Song bigger than video, discarding part of song at end")
        song_points_f = song_points_f[0:int(len(video_points_f))]

    m_sim = evaluator.metrics.calc_emosim(song_points_f, video_points_f)
    print("Mean similarity for "+opt_method+": "+str(m_sim))

    return song_points_f, video_points_f
