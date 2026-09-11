# DiCaprio demo

Scores three genuine test clips and four deepfakes against eight genuine
reference clips. Produces scores and **Real 2 / Fake 2** as an MP4 and looping GIF.

Start with the [CUPID prerequisites](../README.md#install): an NVIDIA GPU and a
CUDA-enabled Python environment are required; there is no CPU fallback.
The demo additionally needs **Bash, the FFmpeg executable with libx264, and
DejaVu Sans fonts**. Run from the repository root:

```bash
python -m pip install -e '.[demo]'
cd demo
./download.sh
./prepare_clips.sh
python quickstart.py
```

Downloads use Firefox cookies by default. Set `COOKIES_FROM_BROWSER=chrome`
or `COOKIES_FROM_BROWSER=none` for Chrome or anonymous access. YouTube/TikTok
may block downloads; source URLs are in [links.txt](links.txt).

Already have the prepared clips?

```bash
python quickstart.py --clips path/to/clips --out results --device cuda:0
```

These are the only demo options. Sampling stays fixed at 15 equispaced frames
per video at native resolution. Scoring uses the public `CupidPipeline` API.

## Outputs

Saved in **`demo/results/`** by default, or the directory supplied with `--out`:

- `scores.json`: all seven scores, AUC, margin and sampling settings.
- `comparison.mp4`: Real 2 / Fake 2, 4.5 seconds, 1280×1180 at 24 fps.
- `comparison.gif`: the same comparison, looping, 768×708 at 12 fps.
- `metadata.json`: compact source/model and rendering configuration.

Intermediate frames and maps are removed automatically, including on errors.
Existing unrelated results are not deleted. Use a fresh output directory for
an uncluttered run. This small demonstration is not a generalization benchmark.

To render the comparison without rescoring:

```bash
python overlay_video.py --clips path/to/clips --out results --device cuda:0
```

## Interpretability overlay

The overlay replaces the paper's video-wide averaging with a centered
five-frame average of decoded UV maps. After subtracting the fixed reference
baseline and taking the absolute value, each interpretability map is projected
onto the current frame's face. Both clips share one fixed color range.
The maps are not manipulation probabilities. See the
[paper](https://arxiv.org/abs/2606.20302) for the underlying method.
