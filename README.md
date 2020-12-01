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

Audio-reactive interpolations are specified by a set of functions which generate latents, noise, network bends, model rewrites, and truncation.

These functions are given to the generate() function in generate_audiovisual.py and then rendered by render.py.

Examples demonstrating some of the techniques discussed in the paper can be found in audioreactive/examples/.

Here are the functions that can specified:
```
# args contains all the arguments to generate()/generate_audiovisual.py as well as:
#   audio: raw audio signal
#   sr: sampling rate of audio
#   n_frames: total number of interpolation frames
#   duration: length of audio in seconds

def initialize(args):
    # intialize values used in multiple of the following functions here
    # e.g. onsets, chroma, RMS, segmentations, bpms, etc.
    # this is useful to prevent duplicate computations (get_noise is called for each noise size)
    # remember to store them back in args (e.g. args.onsets = ar.onsets(...))
    return args

def get_latents(selection, args):
    # selection holds some latent vectors generated randomly or from a file (according to what you specify)
    # generate an audioreactive latent tensor of shape [n_frames, layers, latent_dim]
    return latents

def get_noise(height, width, scale, num_scales, args):
    # height and width are the spatial dimensions of the current noise layer
    # scale is the index and num_scales the total number of noise layers
    # generate an audioreactive noise tensor of shape [n_frames, 1, height, width]
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
    #     "modulation": interpolation frame dependent modulation, shape=(n_frames, ...), 
    #     "transform": function that takes a batch of modulation and returns a torch.nn.Module
    #                  that applies the transformation (given the modulation batch),
    # }
    # (The second one is technical debt in a nutshell. It's a workaround to get kornia transforms
    #  to play nicely. You're probably better off using the first option with a th.nn.Module that
    #  has it's modulation as a parameter and keeping count of which frame you're at internally
    #  if you want time-dependent network bends).
    return bends

def get_rewrites(args):
    # generate a dictionary specifying model rewrites
    # each key value pair should follow:
    #       param_name -> [transform, modulation]
    # where: param_name is the fully-qualified parameter name (see generator.named_children())
    #        transfrom & modulation follow the form of the second network bending dict option
    return rewrites

def get_truncation(args):
    # generate a sequence of truncation values of shape (n_frames,)
    return truncation
```

