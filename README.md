# CVPR-NTIRE UGC Video Enhancement Challenge 2025

[![Page](https://img.shields.io/badge/Challenge-Page-darkgreen)](https://codabench.org/competitions/4973/)
[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2505.03007)
[![Challenges](https://img.shields.io/badge/Challenges-NTIRE%202025-orange)](https://cvlai.net/ntire/2025/)
[![Benchmarks](https://img.shields.io/badge/Benchmarks-VideoProcessing-purple)](https://videoprocessing.ai/benchmarks/)
[![Subjective](https://img.shields.io/badge/Subjectify-Comparisons-blue)](https://subjectify.us/)

![ntrie](https://github.com/user-attachments/assets/dd2e77df-9f1b-4aa9-ae7b-81f265bad35b)

Jointly with the NTIRE workshop, we present the first NTIRE challenge on UGC Video Enhancement. This challenge focuses on the development of algorithms to enhance the visual quality of user-generated videos (UGC), which often suffer from various artifacts such as blurring, noise, shaking, compression artifacts, and faded colors. The task is to create methods that not only improve the perceptual quality of these videos but also ensure that the enhanced results retain high visual quality after being recompressed using x265 at 3000 kbps, making the challenge both practical and impactful.

## Dataset

[![Dataset](https://img.shields.io/badge/Dataset-Google%20Drive-brightgreen)](https://drive.google.com/drive/folders/1F26UwGJ5ykrNxxvEoVtKh3-Qrw_YldOE?usp=sharing)  

We provide the original dataset of 150 UGC videos with natural distortions. Only 120 videos were available to participants, 30 were hidden until the end of the competition.

As well, we [release](https://drive.google.com/drive/folders/1F26UwGJ5ykrNxxvEoVtKh3-Qrw_YldOE?usp=sharing) all the results of processing these 150 videos using the challenge participants' methods (the results of methods on 30 private videos were obtained by the organizers' team independently running the participants' code). All videos were postprocessed by x265 transcoding at 3000kb and were shown to the crowd-sourcing assessors during subjective comparison in exactly the same form. Transcoding command: 

```ffmpeg -i input_path -c:v libx265 -preset fast -b:v 3000k -pix_fmt yuv420p -an output_path```


## Evaluation

To evaluate the videos at all phases of the competition, we largely relied on the [Subjectify.us](https://subjectify.us/) platform. In total, more than 8,000 crowd-sourced assessors took part in the comparisons. Assessors were asked 20 questions in a side-by-side mode, with a random 2 questions serving as verification questions with a known answer. The performers knew nothing about the methods being compared.  

Only votes from performers who passed both verification questions were selected. Then the matrix of pairwise votes was randomly balanced so that each pair had exactly 10 votes. The obtained votes are presented in the ```subjective_votes``` folder. 

To obtain rank scores from pairwise votes, we used the Bradley-Terry model. This transformation is given in ```Subjective.ipynb```, the resulting scores are stored in ```subjective_scores``` folder. In addition, we compute 95% confidence intervals w.r.t. original video.

Please follow the versions in ```requirements.txt``` for reproduction of our results.
