params = dict(
    # parameters for mel-spectrograms (don't change) 
    fftSize = 1024,
    windowSize = 1024,
    hopSize = 256,  
    melBands = 100,
    sampleRate = 24000,
    fmin = 0,
    fmax = 12000,
    melTrainWindow = 64,  # The shape of training audios = batch * 1 * (melTrainWindow * hopSize)
    
    # parameters for training
    trainInitialLR = 7.5e-5,
    trainFinalLR = 5e-6,
    trainBetas = (0.9, 0.99),
    trainWeightDecay = 5e-4,
    trainSteps = 1000000,
    trainBatch = 24,
    trainCheckPointSavingStep = 10000,
    trainAudiosPath = "Your Audios",
    trainMelsPath = "Your Mels",
    trainCheckPointPath = "./checkpoints/WaveFM_0",
    trainGPUs = [0], # [] represents CPU
    
    # parameters for distillation
    distillInitialLR = 2e-5,
    distillFinalLR = 5e-6,
    distillBetas = (0.8, 0.95),
    distillWeightDecay = 1e-2,
    distillDeltaT = 0.01,
    distillSteps = 25000,
    distillBatch = 24,
    distillCheckPointSavingStep = 5000,
    distillModelPath = "./checkpoints/WaveFM_1000000",
    distillCheckPointPath = "./checkpoints/Distilled_WaveFM_0",
    distillAudiosPath = "Your Audios",
    distillMelsPath = "Your Mels",
    distillGPUs = [0], # [] represents CPU
    
    # parameters for inference
    inferenceSteps = 6,
    inferenceMelsPath = "S:/WaveFM2/src/libridev",
    inferenceSavingPath = "S:/dataset/0WaveODE/LIBRIDEV",
    inferenceCheckPointPath = "./checkpoints/WaveFM_1000000", 
                             #"./checkpoints/Distilled_WaveFM_25000",
    inferenceWithGPU = True, # use cuda:0 or CPU
    
    # parameters for model
    timeEmbeddingSize = 512,
    velocityChannels = [512, 256, 128, 64, 32],
    velocityUpSampleRates = [8, 8, 2, 2],
    velocityKernelSizesUp = [[3, 7, 11], [3, 7, 11], [3, 7, 11], [3, 7, 11]],
    velocityDilationsUp = [[1, 3, 5], [1, 3, 5], [1, 3, 5], [1, 3, 5]],
    velocityKernelSizesDown = [[3], [3], [3], [3]],
    velocityDilationsDown = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
)
