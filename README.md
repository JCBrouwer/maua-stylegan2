# maua-stylegan2

This is the repo for my experiments with StyleGAN2. There are many like it, but this one is mine.

It contains the code for [Audio-reactive Latent Interpolations with StyleGAN](https://wavefunk.xyz/assets/audio-reactive-stylegan/paper.pdf) for the NeurIPS 2020 [Workshop on Machine Learning for Creativity and Design](https://neurips2020creativity.github.io/).

The original base is [Kim Seonghyeon's excellent implementation](https://github.com/rosinality/stylegan2-pytorch), but I've gathered code from multiple different repositories or other places online and hacked/grafted it all together. License information for the code should all be in the LICENSE folder, but if you find anything missing or incorrect please let me know and I'll fix it immediately. Tread carefully when trying to distribute any code from this repo, it's meant for research and demonstration.

The files/folders of interest and their purpose are:

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

## Installation

```bash
git clone https://github.com/JCBrouwer/maua-stylegan2
cd maua-stylegan2
pip install -r requirements.txt
```

Alternatively, check out this [Colab Notebook](https://colab.research.google.com/drive/1Ig1EXfmBC01qik11Q32P0ZffFtNipiBR)

## Generating audio-reactive interpolations

The simplest way to get started is to try either (in shell):
```bash
python generate_audiovisual.py --ckpt "/path/to/model.pt" --audio_file "/path/to/audio.wav"
```
or (in e.g. a jupyter notebook):
```python
from generate_audiovisual import generate
generate("/path/to/model.pt", "/path/to/audio.wav")
```

This will use the default audio-reactive settings (which aren't great).

To customize the generated interpolation, more functions can be defined to generate latents, noise, network bends, model rewrites, and truncation.

```python
import audioreactive as ar
from generate_audiovisual import generate

def initialize(args):
    args.onsets = ar.onsets(args.audio, args.sr, ...)
    args.chroma = ar.chroma(args.audio, args.sr, ...)
    return args

def get_latents(selection, args):
    latents = ar.chroma_weight_latents(args.chroma, selection)
    return latents

def get_noise(height, width, scale, num_scales, args):
    noise = ar.perlin_noise(...)
    noise *= 1 + args.onsets
    return noise

generate(ckpt="/path/to/model.pt", audio_file="/path/to/audio.wav", initialize=initialize, get_latents=get_latents, get_noise=get_noise)
```

When running from command line, the `generate()` call at the end can be left out and the interpolation can be generated with:

```bash
python generate_audiovisual.py --ckpt "/path/to/model.pt" --audio_file "/path/to/audio.wav" --audioreactive_file "/path/to/the/code_above.py"
```

This lets you change arguments on the command line rather than having to add them to the `generate()` call in you python file (use whatever you prefer).

Within these functions, you can execute any python code to make the inputs to the network react to the music. There are a number of useful functions provided in `audioreactive/` (imported above as `ar`).

Examples showing how to use the library and demonstrating some of the techniques discussed in the paper can be found in `audioreactive/examples/`. A playlist with example results can be found [here](https://www.youtube.com/watch?v=2LxHRGppdpA&list=PLkain1QGMwiWndQwr3U4shvNpoFC21E3a).

One important thing to note is that the outputs of the functions must adhere strictly to the expected formats. 

Each of the functions is called with all of the arguments from the command line (or `generate()`) in the `args` variable. On top of the arguments, `args` also contains:
- audio: raw audio signal
- sr: sampling rate of audio
- n_frames: total number of interpolation frames
- duration: length of audio in seconds

```python
def initialize(args):
    # intialize values used in multiple of the following functions here
    # e.g. onsets, chroma, RMS, segmentations, bpms, etc.
    # this is useful to prevent duplicate computations (get_noise is called for each noise size)
    # remember to store them back in args
    ...
    return args

def get_latents(selection, args):
    # selection holds some latent vectors (generated randomly or from a file)
    # generate an audioreactive latent tensor of shape [n_frames, layers, latent_dim]
    ...
    return latents

def get_noise(height, width, scale, num_scales, args):
    # height and width are the spatial dimensions of the current noise layer
    # scale is the index and num_scales the total number of noise layers
    # generate an audioreactive noise tensor of shape [n_frames, 1, height, width]
    ...
    return noise

def get_bends(args):
    # generate a list of dictionaries specifying network bends
    # these must follow one of two forms:
    #
    # either: {
    #     "layer": layer index to apply bend to,
    #     "transform": torch.nn.Module that applies the transformation,
    # }
    # or: {
    #     "layer": layer index to apply bend to,
    #     "modulation": time dependent modulation of the transformation, shape=(n_frames, ...), 
    #     "transform": function that takes a batch of modulation and returns a torch.nn.Module
    #                  that applies the transformation (given the modulation batch),
    # }
    # (The second one is technical debt in a nutshell. It's a workaround to get kornia transforms
    #  to play nicely. You're probably better off using the first option with a th.nn.Module that
    #  has its modulation as an attribute and keeps count of which frame it's rendering internally).
    ...
    return bends

def get_rewrites(args):
    # generate a dictionary specifying model rewrites
    # each key value pair should follow:
    #       param_name -> [transform, modulation]
    # where: param_name is the fully-qualified parameter name (see generator.named_children())
    #        transform & modulation follow the form of the second network bending dict option above
    ...
    return rewrites

def get_truncation(args):
    # generate a sequence of truncation values of shape (n_frames,)
    ...
    return truncation
```

The arguments to `generate_audiovisual.py` are as follows. The first two are required, and the remaining are optional.
```bash
generate_audiovisual.py
  --ckpt CKPT                              # path to model checkpoint
  --audio_file AUDIO_FILE                  # path to audio file to react to
  --audioreactive_file AUDIOREACTIVE_FILE  # file with audio-reactive functions defined (as above)
  --output_dir OUTPUT_DIR                  # path to output dir
  --offset OFFSET                          # starting time in audio in seconds (defaults to 0)
  --duration DURATION                      # duration of interpolation to generate in seconds (leave empty for length of audiofile)
  --latent_file LATENT_FILE                # path to latents saved as numpy array
  --shuffle_latents                        # whether to shuffle the supplied latents or not
  --out_size OUT_SIZE                      # ouput video size: [512, 1024, or 1920]
  --fps FPS                                # output video framerate
  --batch BATCH                            # batch size to render with
  --truncation TRUNCATION                  # truncation to render with (leave empty if get_truncations() is in --audioreactive_file)
  --randomize_noise                        # whether to randomize noise
  --dataparallel                           # whether to use data parallel rendering
  --stylegan1                              # if the model checkpoint is StyleGAN1
  --G_res G_RES                            # training resolution of the generator
  --base_res_factor BASE_RES_FACTOR        # factor to increase generator noise maps by (useful when e.g. doubling 512px net to 1024px)
  --noconst                                # whether the generator was trained without a constant input layer
  --latent_dim LATENT_DIM                  # latent vector size of the generator
  --n_mlp N_MLP                            # number of mapping network layers
  --channel_multiplier CHANNEL_MULTIPLIER  # generator's channel scaling multiplier
```

Alternatively, `generate()` can be called directly from python. It takes the same arguments as generate_audiovisual.py except instead of supplying an audioreactive_file, the functions should be supplied directly (i.e. initialize, get_latents, get_noise, get_bends, get_rewrites, and get_truncation as arguments).

Model checkpoints can be converted from tensorflow .pkl's with [Kim Seonghyeon's script](https://github.com/rosinality/stylegan2-pytorch/blob/master/convert_weight.py) (the one in this repo is broken). Both StyleGAN2 and StyleGAN2-ADA tensorflow checkpoints should work once converted. A good place to find models is [this repo](https://github.com/justinpinkney/awesome-pretrained-stylegan2).

There is minimal support for rendering with StyleGAN1 checkpoints as well, although only with latent and noise (no network bending or model rewriting).

## Citation

If you use the techniques introduced in the paper or the code in this repository for your research, please cite the paper:
```
@InProceedings{Brouwer_2020_NeurIPS_Workshops},
    author = {Brouwer, Hans},
    title = {Audio-reactive Latent Interpolations with StyleGAN},
    booktitle = {Proceedings of the 4th Workshop on Machine Learning for Creativity and Design at NeurIPS 2020},
    month = {December},
    year = {2020},
    url={https://jcbrouwer.github.io/assets/audio-reactive-stylegan/paper.pdf}
}
```
