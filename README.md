# Musical Hyperlapse SIBGRAPI 2021 Code #

This project contains the code of the paper published in sibgrapi 2021: [Musical Hyperlapse - A Multimodal Approach to Accelerate First Person Videos](http://sibgrapi.sid.inpe.br/rep/8JMKD3MGPEW34M/45CS7CS). It implements a video acceleration method that considers the visual content of the video and the acoustic content of the background music (Musical Hyperlapse).

## Contact ##

### Authors ###

* Diognei de Matos - MsC student - UFMG - diogneimatos@dcc.ufmg.br
* Washington Luis de Souza Ramos - PhD student - UFMG - washington.ramos@outlook.com
* Luiz Henrique Romanhol - Undergraduate student - UFMG - luizromanhol@dcc.ufmg.br
* Erickson Rangel do Nascimento - Advisor - UFMG - erickson@dcc.ufmg.br

### Institution ###

Federal University of Minas Gerais (UFMG)  
Computer Science Department  
Belo Horizonte - Minas Gerais - Brazil 

### Laboratory ###

**VeRLab:** Laboratory of Computer Vison and Robotics   
https://www.verlab.dcc.ufmg.br

## Code ##

### Dependencies

The following libraries are required to run this project:

- [dijkstra](https://pypi.org/project/dijkstra)
- [dtw](https://pypi.org/project/dtw)
- [essentia](https://pypi.org/project/essentia)
- [ffmpeg-python](https://pypi.org/project/ffmpeg-python)
- [librosa](https://pypi.org/project/librosa)
- [matplotlib](https://pypi.org/project/matplotlib)
- [numpy](https://pypi.org/project/numpy)
- [opencv-python](https://pypi.org/project/opencv-python)
- [pandas](https://pypi.org/project/pandas)
- [scikit-image](https://pypi.org/project/scikit-image)
- [torch](https://pypi.org/project/torch)
- [torchvision](https://pypi.org/project/torchvision)

### Directories ###

You can set the directories in the header.py file. The directories names are stored in the following constants:

* audio_dataset_dir: Directory containning the audio dataset (DEAM), used to train the music emotion neural networks.
  * This directory also contains the songs used to run the final experiments (MSHP).
* image_dataset_dir: Directory containning the image dataset (MVSO), used to train the image emotion neural networks. 
* video_dataset_dir: Directory containning the video dataset (MSHP), used to run the video experiments.
* saved_models_dir: Directory containning the trainned models.
* cache_dir: Directory where cache files will be stored during the code execution.
 
Note: If different users are running the code on the same machine, the cache_dir directory must be different for each user, to avoid conflicts during executions. However, even if this directory is the same for everyone, the code will still run normally. The other directories can remain the same for all users.

### Running ###

In the code is possible to perform several different tasks using the -t flag. Task types are listed below.

| Task                       | Description                                           | Additional Parameters    |
| -------------------------- | ------------------------------------------------------|--------------------------|
| clean                      | Remove the model defined in the header file           | -n                       |
| train                      | Train the model defined in the header file            | -n                       |
| test                       | Test the model defined in the header file             | -n                       |
| save_best                  | Copy the selected model to the saved models directory | -n                       |
| prepare_deam               | Prepare the deam dataset for use                      | None                     |
| prepare_mvso               | Prepare the mvso dataset for use                      | None                     |
| predict_audio              | Create the music emotion curves for an input song     | -s, -r                   |
| predict_video              | Create the video emotion curves for an input video    | -v, -r                   |
| predict_images             | Create a video emotion curve for a list of images     | -l, -r                   |
| make_hyperlapse            | Create a hyperlapse video from a pair <video,song>    | -v, -s, -m, -r           |
| run_experiments            | Calculate all metrics for a video                     | -v,                      |
| create_full_table          | Create a table with all results for all videos        | None                     |
| crop_results               | Generate video pieces of 10 secons from the results   | None                     |
| run_basecomp               | Calculate the metrics for the hyperlapse baselines    | -v, -m                   |

The main tasks are make_hyperlapse and run_experiments.

To make a hyperlapse, you need to specify an input video (-v), an input song or a directory with multiple songs (-s), the acceleration method to use (-m [ours, dtw or greedy]), and whether to generate a video animation (-r [0 or 1]).

Examples:
  ```bash
  python3 main.py -t make_hyperlapse -v input_video.mp4 -s input_song.mp3 -m ours -r 1
  python3 main.py -t make_hyperlapse -v input_video.mp4 -s input_songs/ -m dtw -r 0
  ```
 
To run the experiments for a video in the video dataset dir, you need to specify an inpud video (-v). The songs are fixed in the "<audio_dataset_dir>/MSHP2/" and you can change it by editing the "songs_fix.csv" file mannualy.

Exemples:
  ```bash
  python3 main.py -t run_experiments -v input_video.mp4
  ```

For the run_basecomp tasks, the results for the MSH, SASv1 and SASv2 baselines need to be previous generated externaly, and the directory containning these results need to be defined in the basecomp.py file. Prease see the code for more details.

There are other tasks that can be performed. For more details, please see the main.py file.

Other Examples:
  ```bash
  python3 main.py -t predict_audio -s input_song.mp3 -r 1
  python3 main.py -t predict_video -v input_video.mp4 -r 1
  ```

For each task type, a different set of parameters must be passed. Possible parameters to be passed are listed below.

| Parameter | Description              | Type   | Values            | Default | Example         |
|-----------|--------------------------|--------|-------------------|---------|-----------------|
| -n        | Model version name       | String | -                 | random  | video_model_v1  |
| -s        | Input song               | String | -                 | None    | input_song.mp3  |
| -v        | Input video              | String | -                 | None    | input_video.mp4 |
| -l        | Input list               | String | -                 | None    | input_list.csv  |
| -m        | Frame selection method   | String | ours, dtw, greedy | None    | ours            |
| -r        | Video renderization flag | String | 0 , 1             | 0       | 1               |

For more parameter details, please see the argparser configuration in header.py file.

### Cache Files ###

The cache_dir contains cached files in order to make the algorithm run faster by generating files that only need to be generated once. The following is a description of the files stored in this directory. The experiment results also are stored in this directory.

| Subdirectory    | Description                                                                            |
|-----------------|----------------------------------------------------------------------------------------|
| audio           | Temporary song processing files, such as labels and animation plots                    |   
| video           | Temporary video processing files, such as labels, extracted frames and animation plots |
| combiner        | Temporary files used when creating the results combining a video with a song           |
| evaluator       | Results for the comparison of our method and the DTW and Greedy baselines              |
| basecomp        | Results for the comparison of our method and the MSH, SASv1 and SASv2 baselines        |

Note that the results of the run_experiments and run_basecomp tasks are stored in the directories <cache_dir>/evaluator/ and <cache_dir>/basecomp/ for the MusicalHyperlapse baselines and Hyperlapse baselines, respectively. We recommend not deleting this directory, as it took one week to generate the cache files for all songs, and two weeks to generate the results for 20 videos x 10 songs.
