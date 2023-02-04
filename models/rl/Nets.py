import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    device = torch.device("cuda")

    def __init__(self, inputSize, hiddenSize):
        super(Encoder, self).__init__()
        self.hiddenSize = hiddenSize
        self.gru = nn.GRU(inputSize, self.hiddenSize, batch_first=True).to(self.device)

    def forward(self, input, h0):
        self.gru.flatten_parameters()
        output, hn = self.gru(input, h0)
        return output, hn

    def initHidden(self, batchSize):
        self.batchSize = batchSize
        return torch.zeros(1, batchSize, self.hiddenSize, device=self.device)


class DecoderAttnFull(nn.Module):
    device = torch.device("cuda")

    def __init__(self, maxSeqLen, statusSize, hiddenSize, outputSize, dropout_p):
        super(DecoderAttnFull, self).__init__()
        self.maxSeqLen = maxSeqLen
        self.dropout_p = dropout_p
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.attn = nn.Sequential(
            nn.Linear(self.hiddenSize * 2 + statusSize, 512),
            nn.LeakyReLU(0.5),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.5),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.5),
            nn.Linear(256, self.hiddenSize)
        ).to(self.device)
        self.attn_combine = nn.Sequential(
            nn.Linear(self.hiddenSize * 2, 512),
            nn.LeakyReLU(0.5),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.5),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.5),
            nn.Linear(256, self.hiddenSize)
        ).to(self.device)
        self.attn_normlise = nn.BatchNorm1d(self.hiddenSize * 2)
        self.gru = nn.GRU(self.hiddenSize, self.hiddenSize, batch_first=True).to(self.device)
        self.out = nn.Linear(self.hiddenSize, self.outputSize).to(self.device)

    def forward(self, encoderOutput, hn, status):
        # concat the hidden
        concatVec = torch.cat((encoderOutput, hn.squeeze(0), status), dim=1)
        attn_weights = F.softmax(self.attn(concatVec), dim=1)
        attn_applied = torch.mul(attn_weights, encoderOutput)

        concatAttnApplied = torch.cat((encoderOutput, attn_applied), dim=1)
        concatAttnApplied_normalised = self.attn_normlise(concatAttnApplied)
        output = self.attn_combine(concatAttnApplied_normalised)

        output = F.relu(output).unsqueeze(1)
        self.gru.flatten_parameters()
        output, hn = self.gru(output, hn)  # (N, L, H_in)

        output = self.out(output).squeeze(1)
        return output, hn, attn_weights

    def initHidden(self, batchSize):
        self.batchSize = batchSize
        return torch.zeros(1, batchSize, self.hiddenSize, device=self.device)


class AttentionTimeSeries(nn.Module):
    def __init__(self, hiddenSize, inputSize, seqLen, batchSize, outputSize, statusSize, pdrop):
        super(AttentionTimeSeries, self).__init__()
        self.hiddenSize = hiddenSize
        self.inputSize = inputSize
        self.seqLen = seqLen
        self.batchSize = batchSize
        self.outputSize = outputSize
        self.encoder = Encoder(inputSize, hiddenSize)
        self.decoder = DecoderAttnFull(seqLen, statusSize, hiddenSize, outputSize, pdrop)

    def forward(self, state):
        """
        :param state(dict):     'encoderInput': torch tensor (N, L * 2, H_in)
                                'status': torch tensor (N, 2) which are earning and havePosition
        :return: action array
        """
        # start = time.time()
        cur_batchSize = state['encoderInput'].shape[0]
        encoderHn = self.encoder.initHidden(cur_batchSize)
        encoderOutput, encoderHn = self.encoder(state['encoderInput'], encoderHn)

        # assign encoder hidden to decoder hidden
        decoderHn = encoderHn
        decoderOutput, decoderHn, attn_weights = self.decoder(encoderOutput[:, -1, :], decoderHn, state['status'].to(self.encoder.device))
        # print(time.time() - start)
        return decoderOutput

# ===========================================================================
class SimpleLSTM(nn.Module):
    def __init__(self, inputSize, hiddenSize, lstmLayer, actions_n):
        """
        :param inputSize: int, 6, eg: ohlcvs
        :param hiddenSize: int, eg: 32
        :param lstmLayer: int, eg: 2
        :param actions_n: int, 3 = (B, S, N)
        """
        super(SimpleLSTM, self).__init__()
        self.device = torch.device("cuda")

        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.lstmLayer = lstmLayer

        self.lstm = nn.LSTM(self.inputSize, self.hiddenSize, self.lstmLayer, batch_first=True)

        self.fc_val = nn.Sequential(
            nn.Linear(self.hiddenSize, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        ).to(self.device)

        self.fc_adv = nn.Sequential(
            nn.Linear(self.hiddenSize, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, actions_n),
        ).to(self.device)

    def init_hidden(self, batchSize):
        weight = next(self.parameters()).data
        hidden = (weight.new(batchSize, self.lstmLayer, self.hiddenSize).zero_().cuda(),
                  weight.new(batchSize, self.lstmLayer, self.hiddenSize).zero_().cuda())
        return hidden

    def forward(self, x, hidden):
        """
        :param x: input, eg ohlcvs
        :param hidden: always init, mean=0, std=1
        :return: action vector
        """
        x = x.to(self.device)
        out, hidden = self.lstm(x, hidden)
        val = self.fc_val(out)
        adv = self.fc_adv(out)
        return val + adv - adv.mean(dim=1, keepdim=True)

# ===========================================================================
class SimpleFFDQN(nn.Module):
    def __init__(self, obs_len, actions_n):
        super(SimpleFFDQN, self).__init__()
        self.device = torch.device("cuda")

        self.fc_val = nn.Sequential(
            nn.Linear(obs_len, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        ).to(self.device)

        self.fc_adv = nn.Sequential(
            nn.Linear(obs_len, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, actions_n),
        ).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        val = self.fc_val(x)
        adv = self.fc_adv(x)
        return val + adv - adv.mean(dim=1, keepdim=True)
# ===================== TEST =====================

# hiddenSize = 128
# inputSize = 57
# seqLen = 30
# batchSize = 32
# outputSize = 3
# statusSize = 2
#
# encoderInput = torch.randn(batchSize, seqLen, inputSize)
#
# attentionTimeSeries = AttentionTimeSeries(hiddenSize=hiddenSize, inputSize=inputSize, seqLen=seqLen, batchSize=batchSize, outputSize=outputSize, statusSize=statusSize, pdrop=0.1)
# outputAction = attentionTimeSeries({'encoderInput': encoderInput.to(torch.device("cuda")),
#                                     'status': torch.randn(batchSize, 2).to(torch.device("cuda"))})

# ===================== TEST =====================

# target_model = copy.deepcopy(attentionTimeSeries)
# target_model.load_state_dict(attentionTimeSeries.state_dict())
# print()
