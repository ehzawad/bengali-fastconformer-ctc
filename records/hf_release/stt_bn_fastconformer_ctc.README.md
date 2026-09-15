---
license: cc-by-sa-4.0
language:
- bn
library_name: nemo
pipeline_tag: automatic-speech-recognition
tags:
- ASR
- Bengali
- Bangla
- FastConformer
- CTC
- NeMo
base_model: nvidia/stt_en_fastconformer_ctc_large
---

# stt_bn_fastconformer_ctc

Bengali speech-to-text, FastConformer-CTC large (115.6 M parameters, 8× subsampling), greedy CTC decode,
no language model, 1,024-piece Bengali BPE, trained on 1,692 hours of human-supervised Bengali.

Weights: `stt_bn_fastconformer_ctc.nemo`, SHA-256 `bea300ce4f1f1df1fa4707562ddaa7c64db6f01121ef3d64a6333635f3626572`.

## Output

Input 16 kHz mono WAV/FLAC; output normalised Bengali text without punctuation.

```python
# pip install -q 'nemo_toolkit[asr]==2.7.3'
import nemo.collections.asr as nemo_asr
m = nemo_asr.models.ASRModel.from_pretrained("ehzawad/stt_bn_fastconformer_ctc")
print(m.transcribe(["your_audio_16k_mono.wav"])[0].text)
```

## Dataset

1,084,696 clips, **1,692.4 hours**, all human-read or human-transcribed; no pseudo-labels, no synthetic audio.

| Source | Hours | Licence |
|---|---:|---|
| [Bengali.AI Speech](https://www.kaggle.com/competitions/bengaliai-speech) (Kaggle `bengaliai-speech`, train split, filtered by [tugstugi](https://github.com/tugstugi)'s WER < 15 id list; transcripts are the competition's own) | 726.5 | CC-BY-4.0 |
| [IndicVoices](https://huggingface.co/datasets/ai4bharat/IndicVoices) (Bengali) | 574.4 | CC-BY-4.0 |
| [OpenSLR-53](https://www.openslr.org/53/) | 159.0 | CC-BY-SA-4.0 |
| [SUBAK.KO](https://huggingface.co/datasets/SUST-CSE-Speech/SUBAK.KO) | 105.8 | CC-BY-4.0 |
| [Kathbath](https://huggingface.co/datasets/ai4bharat/kathbath) (Bengali) | 81.5 | CC-BY-4.0 |
| [Common Voice](https://commonvoice.mozilla.org/bn) (validated) | 39.5 | CC0 |
| [FLEURS](https://huggingface.co/datasets/google/fleurs) `bn_in` train split | 5.8 | CC-BY-4.0 |

Base checkpoint: [`nvidia/stt_en_fastconformer_ctc_large`](https://huggingface.co/nvidia/stt_en_fastconformer_ctc_large), © NVIDIA, CC-BY-4.0.
Full credits, licence notices and modifications: [SUPPLEMENT.md](SUPPLEMENT.md).

## Benchmark

FLEURS Bengali official **test** split, 920 recordings, corpus WER on punctuation-normalised text, greedy
decoding, no language model: **WER 17.12 %, CER 5.12 %**.

Under the same scorer on the same 920 recordings, the public comparison in
[bengali-asr-benchmark](https://github.com/ehzawad/bengali-asr-benchmark): hishab Conformer Large 14.37 %,
Whisper Medium (tugstugi) 16.23 %, this model 17.12 %, wav2vec2 20.64 %, hishab FastConformer 36.22 %. On the
official test splits of corpora none of these models trained on, this model leads: Vaani test 24.66 %,
SPRING-INX R1 39.24 %, R2 38.41 % (1,000-utterance subsets); on Kathbath test it reads 13.4 % against Conformer
Large's 9.3 %; on IndicVoices conversational 12.6 % (it trained on that corpus's train split; the others read 25–33 %).

Held-out panels, the benchmark tables and the limitations: [SUPPLEMENT.md](SUPPLEMENT.md).

## Licence

Weights: [CC-BY-SA-4.0](https://creativecommons.org/licenses/by-sa/4.0/). Attribution and share-alike
follow from OpenSLR-53; the required notices are in [SUPPLEMENT.md](SUPPLEMENT.md).

## Citation

```bibtex
@misc{zawad2026bengalifastconformerctc,
  title        = {Bengali FastConformer-CTC: Expanded-Corpus Speech Recognition Under a Registered Protocol},
  author       = {Zawad, Emrul},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/ehzawad/stt_bn_fastconformer_ctc}}
}
```
