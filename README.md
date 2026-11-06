# 🏒 Jersey & Player-Role Recognition (Ice Hockey Computer Vision)

**Distinction Award · IEYI China (International Exhibition for Young Inventors)** · Grade 12 independent research

A computer-vision system for hockey game film. YOLO detects everyone on the ice, a CNN sorts players into teams by jersey colour, and movement is tracked frame by frame to estimate player speed, distance covered and offensive pressure.

> **Inspo from: [Raúl Vizcarra Chirinos](https://github.com/rvizcarra15)**, whose open-source project
> [IceHockey_ComputerVision_PyTorch](https://github.com/rvizcarra15/IceHockey_ComputerVision_PyTorch)
> and article [Spicing up Ice Hockey with AI](https://towardsdatascience.com/spicing-up-ice-hockey-with-ai-player-tracking-with-computer-vision-ce9ceec9122a)
> this project builds on. The tracking pipeline, rink-zone code and sample datasets come from his work, and his original commits are kept in this repo's history.

## What I added

- **My trained team classifier**: `hockey_team_classifier.pth`, trained on the labelled sample dataset (home team, away team, referee, puck).
- **Puck class**: added a fourth class to the classifier.
- **Runs out of the box**: relative file paths and GPU (CUDA) loading.
- **Chinese comments** throughout the scripts, for my exhibition team.
- **My results**: tracking output and stats video in `demo/`.

## Demo videos

- [▶ demo/ryan-demo.mp4](demo/ryan-demo.mp4): short demo of the model running on game footage
- [▶ demo/ryan-output-video.mp4](demo/ryan-output-video.mp4): full tracked output with team stats overlay

Also on my website: https://ryanli0819.github.io/#project-hockey-cv

## Skills used

PyTorch · CNNs · YOLO · OpenCV · Object tracking · Data labelling

## Results

- **90%+** classification accuracy
- **4** classes tracked (two teams, referee, puck)

## How to run

1. `pip install ultralytics torch torchvision opencv-python matplotlib`
2. Download `yolov8x.pt` from [Ultralytics](https://github.com/ultralytics/assets/releases) and put it in the repo root. It's 137 MB, over GitHub's file size limit, so it isn't included.
3. Unzip `teams_sample_dataset.zip` and `validation_dataset.zip`.
4. `python GH_CV_track_teams_vf.py` reads `video_input.mp4` and writes `output_video.mp4`. It loads the classifier on a CUDA GPU; change `cuda` to `cpu` in `load_classifier` if you don't have one.

---
Made by **Ryan Li** · Data Analytics @ Purdue · [Portfolio](https://ryanli0819.github.io) · [GitHub](https://github.com/ryanli0819)
