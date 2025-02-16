# WaveFM: A High-Fidelity and Efficient Vocoder Based on Flow Matching

Our audio demo page is available at https://luotianze666.github.io/WaveFM.

## Basic Usage
## 1. Get Mel-Spectrograms
You can extract Mel-spectrograms for both the training and inference processes using the following command.
```python
python dataset.py -i your_audio_path -o your_mel_saving_path
```
## 2. Train WaveFM
First modify the paths for `training` in `src/params.py` as needed, and then use `python src/train.py` to train WaveFM.
```python
# params for training
trainInitialLR = 7.5e-5,
trainFinalLR = 5e-6,
trainBetas = (0.9, 0.99),
trainWeightDecay = 5e-4,
trainSteps = 1000000,
trainBatch = 32,
trainCheckPointSavingStep = 10000,
trainAudiosPath = "Your Audios",
trainMelsPath = "Your Mels",
trainCheckPointPath = "./checkpoints/WaveFM_0",
trainGPUs = [0,1], # [] represents CPU
```
## 3. Use Checkpoints For Inference 
First modify the paths for `inference` in `src/params.py` as needed, and then use `python src/inference.py` to run inference for WaveFM.
```python
# params for inference
inferenceSteps = 6,
inferenceMelsPath = "Your Mels For Inference",
inferenceSavingPath = "./generation",
inferenceCheckPointPath = "./checkpoints/WaveFM_1000000", 
                        # "./distilled_checkpoints/Distilled_WaveFM_25000",
inferenceWithGPU = True, # use cuda:0 or CPU
```
## 4. Distill WaveFM
First modify the paths for `distillation` in `src/params.py` as needed, and then use `python src/distillation.py` to distill WaveFM.
```python
# params for distillation
distillInitialLR = 2e-5,
distillFinalLR = 5e-6,
distillBetas = (0.8, 0.95),
distillWeightDecay = 1e-2,
distillDeltaT = 0.01,
distillSteps = 25000,
distillBatch = 32,
distillCheckPointSavingStep = 5000,
distillModelPath = "./checkpoints/WaveFM_1000000",
distillCheckPointPath = "./distilled_checkpoints/Distilled_WaveFMfinal_0",
distillAudiosPath = "Your Audios",
distillMelsPath = "Your Mels",
distillGPUs = [0,1], # [] represents CPU
```

## Package Requirements

WaveFM has been tested on `python 3.9` with the following requirements, and it should also work fine with the latest version.

```python
torch           2.0.1+cu118
torchaudio      2.0.2
```
