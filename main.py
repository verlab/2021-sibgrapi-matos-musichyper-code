#-----------------------------------------------------------------------------------------#
#Quick sending code to github

import os
import sys

if(sys.argv[1]=="-g"):
    os.system('find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf')
    os.system('git add -A')
    os.system('git commit -am "-"')
    os.system('git push')
    exit()

if(sys.argv[1]=="-d"):
    os.system('find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf')
    fnam = sys.argv[2]
    if not(os.path.exists("out/")):
        os.mkdir("out/")
    os.system("scp proc2:~/Files/codet/out/"+str(fnam)+" "+"out/"+str(fnam))
    exit()

#-----------------------------------------------------------------------------------------#

from header import *

#-----------------------------------------------------------------------------------------#

def quick_test1():
    video_points = []
    video_points.append((0,0,0,0,0,"/srv/storage/datasets/Diognei/Cache/video/Verlab1/frames/0.jpg"))
    video_points.append((0,0,0,0,0,"/srv/storage/datasets/Diognei/Cache/video/Verlab1/frames/1.jpg"))
    video_points.append((0,0,0,0,0,"/srv/storage/datasets/Diognei/Cache/video/Verlab1/frames/2.jpg"))
    video_points.append((0,0,0,0,0,"/srv/storage/datasets/Diognei/Cache/video/Verlab1/frames/3.jpg"))
    video_points.append((0,0,0,0,0,"/srv/storage/datasets/Diognei/Cache/video/Verlab1/frames/4.jpg"))
    video_fnam = "/srv/storage/datasets/Diognei/Video/MSHP/Verlab1.mp4"
    evaluator.metrics.calc_fidscore(video_points,video_fnam)   

def quick_test2():
    video_points = []
    video_points.append((0,0,0,0,0,"/srv/storage/datasets/Diognei/Cache/video/Verlab1/frames/0.jpg"))
    video_points.append((0,0,0,0,0,"/srv/storage/datasets/Diognei/Cache/video/Verlab1/frames/1.jpg"))
    video_points.append((0,0,0,0,0,"/srv/storage/datasets/Diognei/Cache/video/Verlab1/frames/2.jpg"))
    video_points.append((0,0,0,0,0,"/srv/storage/datasets/Diognei/Cache/video/Verlab1/frames/3.jpg"))
    video_points.append((0,0,0,0,0,"/srv/storage/datasets/Diognei/Cache/video/Verlab1/frames/4.jpg"))
    stab = evaluator.metrics.calc_vidshak(video_points)  
    print(stab)

#-----------------------------------------------------------------------------------------#
#Main function

def main():

    print("Initializing system...")

    use_gpu = True if(torch.cuda.is_available()) else False
    if not(use_gpu):
        print("Note: GPU not available, will use CPU!")
    
    if(task in ["clean","train","test"]):
        learning.trainer.run(task,version_name,use_gpu)
    elif(task=="save_best"):
        utils.save_model_as_main(version_name)
    elif(task=="prepare_deam"):
        c = utils.str2bool(input("Clean? (y/n) "))
        nc = int(input("Num classes: "))
        ud = utils.str2bool(input("Undersampling: "))
        dataprep.deam.run(num_classes=nc,clean=c,undsamp=ud)
    elif(task=="prepare_mvso"):
        dataprep.mvso.run()
    elif(task=="prepare_other"):
        dataprep.other.run()
    elif(task=="predict_audio"):
        write_video = False if render_video==0 else True
        profgen1 = combiner.profgen.profgen("audio",64,song_filename,write_video,use_gpu)
        profgen1.plot_emotion_profile(render_video)
    elif(task=="predict_video"):
        write_video = False if render_video==0 else True
        profgen1 = combiner.profgen.profgen("video", 4,video_filename,write_video,use_gpu)
        profgen1.plot_emotion_profile(render_video)
    elif(task=="predict_images"):
        write_video = False if render_video==0 else True
        profgen1 = combiner.profgen.profgen("image", 4,list_filename,write_video,use_gpu)
        profgen1.plot_emotion_profile(render_video)
    elif(task=="make_hyperlapse"):
        result_mode = "score" if render_video==0 else "animation"
        opt_method = optim_method
        combiner.hypmaker.make_hyperlapse(video_filename,song_filename,opt_method,result_mode)
    elif(task=="run_experiments"):
        clean_level = int(clean_exps)
        evaluator.runner.run_experiments(video_filename,clean_level)
    elif(task=="create_full_table"):
        evaluator.runner.create_full_table()
    elif(task=="quick_test"):
        quick_test2()
    elif(task=="clean_out_dir"):
        print("Cleaning out dir...")
        if(os.path.exists(out_dir)):
            shutil.rmtree(out_dir)
    elif(task=="run_basecomp"):
        other.basecomp.run(video_filename,optim_method)
    elif(task=="gen_qualplots"):
        evaluator.plotter.run()
    elif(task=="gen_suppmat"):
        other.suppmat.run()
    else:
        print("Undefined task "+str(task))

    utils.clear_pycache()

#-----------------------------------------------------------------------------------------#

if __name__ == "__main__":
    main()

#-----------------------------------------------------------------------------------------#
