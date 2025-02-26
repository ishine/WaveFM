import torch
import torch.nn as nn
import torch.nn.functional as F
from params import params


def filterTime(input):
    input = F.pad(input, pad=(1, 0, 1, 1), mode="constant")
    weight = torch.tensor([[-1.0, 1.0], 
                           [-2.0, 2.0], 
                           [-1.0, 1.0]]).to(input.device).reshape(1, 1, 3, 2) / 4
    deltaT = torch.conv2d(input.unsqueeze(1), weight=weight)
    return deltaT.squeeze(1)


def filterFreq(input):
    input = F.pad(input, pad=(1, 1, 1, 0), mode="constant")
    weight = torch.tensor([[-1.0, -2.0, -1.0], 
                           [1.0, 2.0, 1.0]]).to(input.device).reshape(1, 1, 2, 3) / 4
    deltaF = torch.conv2d(input.unsqueeze(1), weight=weight)
    return deltaF.squeeze(1)


def filterLaplacian(input):
    input = F.pad(input, pad=(1, 1, 1, 1), mode="constant")
    weight = torch.tensor([[-1.0, -1.0, -1.0], 
                           [-1.0, 8.0, -1.0], 
                           [-1.0, -1.0, -1.0]]).to(input.device).reshape(1, 1, 3, 3) / 8
    laplacian = torch.conv2d(input.unsqueeze(1), weight=weight)
    return laplacian.squeeze(1)


def getSTFTLoss(
    answer,
    predict,
    fft_sizes=(1024, 2048, 512),
    hop_sizes=(128, 256, 64),
    win_lengths=(512, 1024, 256),
    window=torch.hann_window,
):
    loss = 0
    for i in range(len(fft_sizes)):
        
        ansStft = torch.view_as_real(
            torch.stft(
                answer.squeeze(1),
                n_fft=fft_sizes[i],
                hop_length=hop_sizes[i],
                win_length=win_lengths[i],
                window=window(win_lengths[i], device=answer.device),
                return_complex=True,
            )
        )
        predStft = torch.view_as_real(
            torch.stft(
                predict.squeeze(1),
                n_fft=fft_sizes[i],
                hop_length=hop_sizes[i],
                win_length=win_lengths[i],
                window=window(win_lengths[i], device=predict.device),
                return_complex=True,
            )
        )
        
        ansStftMag = ansStft[..., 0] ** 2 + ansStft[..., 1] ** 2
        predStftMag = predStft[..., 0] ** 2 + predStft[..., 1] ** 2

        magMin = 1e-6
        mask = (ansStftMag > magMin) & (predStftMag > magMin)

        ansStftMag = torch.sqrt(ansStftMag + magMin)
        predStftMag = torch.sqrt(predStftMag + magMin)

        ansStftPha = torch.atan2(ansStft[..., 1][mask], ansStft[..., 0][mask])
        predStftPha = torch.atan2(predStft[..., 1][mask], predStft[..., 0][mask])

        deltaPhase = ansStftPha - predStftPha
        loss += torch.atan2(torch.sin(deltaPhase), torch.cos(deltaPhase)).abs().mean()
        loss += (ansStftMag.log() - predStftMag.log()).abs().mean()

        ansStftMagDT = filterTime(ansStftMag)
        ansStftMagDF = filterFreq(ansStftMag)
        ansStftMagLap = filterLaplacian(ansStftMag)

        predStftMagDT = filterTime(predStftMag)
        predStftMagDF = filterFreq(predStftMag)
        predStftMagLap = filterLaplacian(predStftMag)

        loss += 4.0 * (ansStftMagDF - predStftMagDF).pow(2).mean()
        loss += 4.0 * (ansStftMagDT - predStftMagDT).pow(2).mean()
        loss += 2.0 * (ansStftMagLap - predStftMagLap).pow(2).mean()

    return loss / len(fft_sizes)


class Snake(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1, channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x):
        res = x.clone()
        res += 1.0 / (self.beta.exp() + 1e-8) * torch.sin(self.alpha.exp() * x).pow(2)
        return res
    
def kaiserSincFilter(cutOff, halfWidth, kernelSize):
    halfSize = kernelSize // 2
    if kernelSize % 2 == 0:
        time = torch.arange(-halfSize, halfSize) + 0.5
    else:
        time = torch.arange(kernelSize) - halfSize

    attenuation = 2.285 * (halfSize - 1) * math.pi * 4 * halfWidth + 7.95

    if attenuation > 50.0:
        beta = 0.1102 * (attenuation - 8.7)
    elif attenuation >= 21.0:
        beta = 0.5842 * (attenuation - 21) ** 0.4 + 0.07886 * (attenuation - 21.0)
    else:
        beta = 0.0
    # Kaiser's estimation for beta
    window = torch.kaiser_window(kernelSize, False, beta)

    filter = (
        2 * cutOff * torch.sinc(2 * cutOff * time) * window
    )  # Kaiser Window with Sinc filter
    if cutOff != 0:
        filter /= filter.sum()
    return filter.view(1, 1, kernelSize)

class LowPassFilter(nn.Module):
    def __init__(self, kernelSize=6, cutOff=0.25, halfWidth=0.6):
        super().__init__()
        self.register_buffer("filter", kaiserSincFilter(cutOff, halfWidth, kernelSize))
        self.padLeft = kernelSize // 2 - int(kernelSize % 2 == 0)
        self.padRight = kernelSize // 2

    def forward(self, x):
        B, C, T = x.shape
        x = F.pad(x, (self.padLeft, self.padRight), "replicate")
        x = F.conv1d(x, self.filter.expand(C, -1, -1), groups=C)
        return x

class DownSampler(nn.Module):
    def __init__(self, ratio=2, kernelSize=12, cutOff=0.5, halfWidth=0.6):
        super().__init__()
        self.register_buffer("filter", kaiserSincFilter(cutOff / ratio, halfWidth / ratio, kernelSize))
        self.padLeft = kernelSize // 2 - int(kernelSize % 2 == 0)
        self.padRight = kernelSize // 2
        self.ratio = ratio

    def forward(self, x):
        B, C, T = x.shape
        x = F.pad(x, (self.padLeft, self.padRight), "replicate")  # Keep the shape
        x = F.conv1d(
            x, self.filter.expand(C, -1, -1), stride=self.ratio, groups=C
        )  # Channel-wise filtering
        return x

class UpSampler(nn.Module):
    def __init__(self, ratio=2, kernelSize=12, cutOff=0.5, halfWidth=0.6):
        super().__init__()
        self.register_buffer("filter", kaiserSincFilter(cutOff / ratio, halfWidth / ratio, kernelSize))
        self.pad = (kernelSize // ratio - 1)  # replicate padding to avoid transposed convolution out of sequence
        self.cropLeft = self.pad * ratio + (kernelSize - ratio) // 2
        self.cropRight = self.pad * ratio + (kernelSize - ratio + 1) // 2
        # Keep the sequence's length correct (L_out = L_in * ratio + kernelSize - ratio)
        self.ratio = ratio

    def forward(self, x):
        B, C, T = x.shape
        x = F.pad(x, (self.pad, self.pad), "replicate")
        x = self.ratio * F.conv_transpose1d(
            x, self.filter.expand(C, -1, -1), stride=self.ratio, groups=C
        )
        x = x[:, :, self.cropLeft : -self.cropRight]
        return x

class AntiAliasingSnake(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.upSampler = UpSampler(ratio=2, kernelSize=12)
        self.downSampler = DownSampler(ratio=2, kernelSize=12)
        self.alpha = nn.Parameter(torch.zeros(1, channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x):
        x = self.upSampler(x)
        res = x.clone()
        res += 1.0 / (self.beta.exp() + 1e-8) * torch.sin(self.alpha.exp() * x).pow(2)
        res = self.downSampler(res)
        return res

class Block(nn.Module):
    def __init__(self, channels, kernelSize=3, dilation=1):
        super().__init__()

        self.conv1 = nn.Conv1d(channels, channels, kernelSize, dilation=dilation, padding="same")
        self.act1 = Snake(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernelSize, padding="same")
        self.act2 = Snake(channels)

    def forward(self, x):
        res = x
        x = self.act1(x)
        x = self.conv1(x)
        x = self.act2(x)
        x = self.conv2(x)
        x += res
        return x


class ResLayer(nn.Module):
    def __init__(self, channels, kernelSize=(3, 5, 7), dilation=(1, 3, 5)):
        super().__init__()
        self.blocks = nn.ModuleList([])
        self.kernelSize = kernelSize
        self.dilation = dilation
        for i in range(len(kernelSize)):
            for j in range(len(dilation)):
                self.blocks.append(Block(channels, kernelSize[i], dilation[j]))

    def forwardOneKernel(self, x, kernelID):
        out = self.blocks[kernelID * len(self.dilation)](x)
        for i in range(1, len(self.dilation)):
            out = self.blocks[kernelID * len(self.dilation) + i](out)
        return out

    def forward(self, x):
        sum = self.forwardOneKernel(x, 0)
        for i in range(1, len(self.kernelSize)):
            sum += self.forwardOneKernel(x, i)
        sum /= len(self.kernelSize)
        return sum

class Velocity(nn.Module):

    @staticmethod
    def timeEmbedding(t):
        if len(t.shape) == 1:
            t = t.unsqueeze(-1)  # batch -> batch*1
        if len(t.shape) == 3:
            t = t.squeeze(-1)  # batch*1*1 -> batch*1

        pos = torch.arange(64, device=t.device).unsqueeze(0)  # 1*64
        table = 100 * t * 10.0 ** (pos * 4.0 / 63.0)  # batch*64

        return torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # batch*128

    def __init__(
        self,
        channels=params["velocityChannels"],
        upSampleRates=params["velocityUpSampleRates"],
        kernelSizesUp=params["velocityKernelSizesUp"],
        dilationsUp=params["velocityDilationsUp"],
        kernelSizesDown=params["velocityKernelSizesDown"],
        dilationsDown=params["velocityDilationsDown"],
    ):
        super().__init__()

        self.timePre0 = nn.Linear(128, params["timeEmbeddingSize"])
        self.timePre1 = nn.Linear(params["timeEmbeddingSize"], params["timeEmbeddingSize"])
        self.SiLU = nn.SiLU()
        self.upSampleRates = upSampleRates

        self.convUpIn = nn.Conv1d(params["melBands"], channels[0], 7, 1, padding="same")
        self.convDownIn = nn.Conv1d(1, channels[-1], 7, padding="same")

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        for i in range(len(upSampleRates)):

            self.ups.append(
                nn.ConvTranspose1d(
                    channels[i],
                    channels[i + 1],
                    kernel_size=2 * upSampleRates[i],
                    stride=upSampleRates[i],
                    padding=upSampleRates[i] // 2,
                )
            )  # stride=2kernel=4padding

            self.downs.append(
                nn.Conv1d(
                    channels[i + 1],
                    channels[i],
                    kernel_size=2 * upSampleRates[i] + 1,
                    stride=upSampleRates[i],
                    padding=upSampleRates[i],
                )
            )

        self.resLayerUps = nn.ModuleList()
        self.resLayerDowns = nn.ModuleList()
        self.timeDowns = nn.ModuleList()

        for i in range(len(upSampleRates)):
            self.timeDowns.append(
                nn.Linear(params["timeEmbeddingSize"], channels[i + 1])
            )
            self.resLayerUps.append(
                ResLayer(channels[i + 1], kernelSizesUp[i], dilationsUp[i])
            )
            self.resLayerDowns.append(
                ResLayer(channels[i + 1], kernelSizesDown[i], dilationsDown[i])
            )

        self.convUpOut = nn.Conv1d(channels[-1], 1, 7, 1, padding="same")
        self.actUpOut = Snake(channels=channels[-1])

  
    def forward(self, x, melSpectrogram, t):

        timeEmbedding = self.timeEmbedding(t)
        timeEmbedding = self.SiLU(self.timePre0(timeEmbedding))
        timeEmbedding = self.SiLU(self.timePre1(timeEmbedding))

        x = self.convDownIn(x)

        skipConnections = [x.clone()]
        for i in range(len(self.downs) - 1, -1, -1):

            x += self.timeDowns[i](timeEmbedding).unsqueeze(-1)
            x = self.resLayerDowns[i](x)
            x = self.downs[i](x)

            skipConnections.append(x.clone())

        melSpectrogram = self.convUpIn(melSpectrogram)
        melSpectrogram += skipConnections[-1]

        for i in range(len(self.ups)):

            melSpectrogram = self.ups[i](melSpectrogram)
            melSpectrogram += skipConnections[-i - 2]
            melSpectrogram = self.resLayerUps[i](melSpectrogram)

        out = self.actUpOut(melSpectrogram)
        out = self.convUpOut(out)
        out = torch.tanh(out)

        return out

