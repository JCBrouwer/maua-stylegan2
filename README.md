This is my fork of stylegan2-pytorch. There are many like it, but this one is mine.

It contains the code for [Audio-reactive Latent Interpolations with StyleGAN](https://wavefunk.xyz/assets/audio-reactive-stylegan/paper.pdf).

The original base is [Rosinality's excellent implementation](https://github.com/rosinality/stylegan2-pytorch), but I've gathered code from multiple different repositories and hacked/grafted it all together. License information for the code should all be in the LICENSE folder, but if you find anything missing or incorrect please let me know and I'll fix it immediately. Tread carefully when trying to distribute any code from this repo, it's meant for research and demonstration

| File/Folder | Description
| :--- | :----------
| generate_audiovisual.py | used to generate audio-reactive interpolations
| audioreactive/ | contains the main functions needed for audioreactiveness + examples demonstrating how they can be used
| render.py | renders interpolations using ffmpeg
| select_latents.py | GUI for selecting latents, left click to add to top set, right click to add to bottom
| models/ | StyleGAN networks
| workspace/ | place to store intermediate results, latents, or inputs, etc.
| output/ | default generated output folder
| train.py | code for training models

The rest of the code is experimental, probably broken, and unsupported.

## Generating audio-reactive interpolations

