"""PyTorch helpers emitted into generated code for sequence and audio blocks.

Sequence layers take batch-first sequences [B, L, C]. Audio front ends take
waveforms [B, 1, samples] (or [B, samples]) and return [B, 1, bins, frames].
Each helper has an analytic twin (``*_cost``) that counts parameters and
multiply-accumulates without importing torch, for the summary and budgets.
"""
from __future__ import annotations

import math

TCN = '''\
class TemporalBlock(nn.Module):
    """Two dilated causal convolutions with a residual connection (Bai et al., 2018)."""

    def __init__(self, cin: int, cout: int, kernel: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(cin, cout, kernel, dilation=dilation)
        self.conv2 = nn.Conv1d(cout, cout, kernel, dilation=dilation)
        self.drop = nn.Dropout(dropout)
        self.down = nn.Conv1d(cin, cout, 1) if cin != cout else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, C, L]
        y = self.drop(torch.relu(self.conv1(nn.functional.pad(x, (self.pad, 0)))))
        y = self.drop(torch.relu(self.conv2(nn.functional.pad(y, (self.pad, 0)))))
        return torch.relu(y + (x if self.down is None else self.down(x)))


class TCN(nn.Module):
    """Temporal convolutional network over [B, L, C] -> [B, L, channels]; dilations 1, 2, 4..."""

    def __init__(self, cin: int, channels: int, levels: int, kernel: int = 3,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*(TemporalBlock(cin if i == 0 else channels, channels,
                                                    kernel, 2 ** i, dropout)
                                      for i in range(levels)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x.transpose(1, 2)).transpose(1, 2)
'''

MAMBA = '''\
import math


class MambaBlock(nn.Module):
    """Selective state-space block (Mamba, Gu & Dao 2023) in plain PyTorch, pre-norm residual.

    [B, L, C] -> [B, L, C]. The selective scan runs as a loop over time steps.
    """

    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2) -> None:
        super().__init__()
        inner = expand * dim
        self.dt_rank = max(1, math.ceil(dim / 16))
        self.d_state = d_state
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, 2 * inner, bias=False)
        self.conv = nn.Conv1d(inner, inner, d_conv, groups=inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, inner)
        a = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(inner, 1)
        self.A_log = nn.Parameter(torch.log(a))
        self.D = nn.Parameter(torch.ones(inner))
        self.out_proj = nn.Linear(inner, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[1]
        u, gate = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        u = nn.functional.silu(self.conv(u.transpose(1, 2))[..., :length].transpose(1, 2))
        dt, b, c = self.x_proj(u).split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = nn.functional.softplus(self.dt_proj(dt))                 # [B, L, E]
        a = -torch.exp(self.A_log.float())                            # [E, N]
        decay = torch.exp(dt.unsqueeze(-1) * a)                       # [B, L, E, N]
        drive = dt.unsqueeze(-1) * b.unsqueeze(2) * u.unsqueeze(-1)   # [B, L, E, N]
        state = torch.zeros_like(decay[:, 0])
        outputs = []
        for t in range(length):
            state = decay[:, t] * state + drive[:, t]
            outputs.append((state * c[:, t].unsqueeze(1)).sum(-1))
        y = torch.stack(outputs, dim=1) + u * self.D
        return x + self.out_proj(y * nn.functional.silu(gate))
'''

S4D = '''\
import math


class S4DLayer(nn.Module):
    """Diagonal structured state-space layer (S4D-Lin, Gu et al. 2022), pre-norm residual.

    The SSM kernel is materialised for the sequence length and applied with an FFT
    convolution: [B, L, C] -> [B, L, C].
    """

    def __init__(self, dim: int, d_state: int = 64, dropout: float = 0.0) -> None:
        super().__init__()
        half = d_state // 2
        self.norm = nn.LayerNorm(dim)
        self.log_dt = nn.Parameter(torch.rand(dim) * (math.log(0.1) - math.log(0.001))
                                   + math.log(0.001))
        self.log_a_real = nn.Parameter(torch.log(0.5 * torch.ones(dim, half)))
        self.a_imag = nn.Parameter(math.pi * torch.arange(half).float().repeat(dim, 1))
        self.c = nn.Parameter(torch.randn(dim, half, 2) * 0.5 ** 0.5)
        self.D = nn.Parameter(torch.randn(dim))
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(dim, dim)

    def kernel(self, length: int) -> torch.Tensor:
        dt = torch.exp(self.log_dt)                                   # [C]
        a = -torch.exp(self.log_a_real) + 1j * self.a_imag            # [C, N]
        c = torch.view_as_complex(self.c) * (torch.exp(dt.unsqueeze(-1) * a) - 1.0) / a
        steps = dt.unsqueeze(-1) * a                                  # [C, N]
        k = c.unsqueeze(-1) * torch.exp(steps.unsqueeze(-1)
                                        * torch.arange(length, device=a.device))
        return 2 * k.sum(1).real                                      # [C, L]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[1]
        u = self.norm(x).transpose(1, 2)                              # [B, C, L]
        k = self.kernel(length)
        y = torch.fft.irfft(torch.fft.rfft(u, n=2 * length) * torch.fft.rfft(k, n=2 * length),
                            n=2 * length)[..., :length]
        y = y + u * self.D.unsqueeze(-1)
        return x + self.out(self.drop(nn.functional.gelu(y.transpose(1, 2))))
'''

SLSTM = '''\
class SLSTM(nn.Module):
    """sLSTM cell stack (xLSTM, Beck et al. 2024): exponential input gates with a
    stabiliser and normaliser state. [B, L, C] -> [B, L, H] (or the final [B, H])."""

    def __init__(self, cin: int, hidden: int, layers: int = 1,
                 return_sequences: bool = True) -> None:
        super().__init__()
        self.hidden, self.return_sequences = hidden, return_sequences
        self.w = nn.ModuleList(nn.Linear(cin if i == 0 else hidden, 4 * hidden)
                               for i in range(layers))
        self.r = nn.ModuleList(nn.Linear(hidden, 4 * hidden, bias=False) for _ in range(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for w, r in zip(self.w, self.r):
            batch, length = x.shape[:2]
            h = x.new_zeros(batch, self.hidden)
            c, n, m = torch.zeros_like(h), torch.ones_like(h), torch.zeros_like(h)
            pre = w(x)
            outputs = []
            for t in range(length):
                i_raw, f_raw, z, o = (pre[:, t] + r(h)).chunk(4, dim=-1)
                log_f = nn.functional.logsigmoid(f_raw)
                m_new = torch.maximum(log_f + m, i_raw)
                i = torch.exp(i_raw - m_new)
                f = torch.exp(log_f + m - m_new)
                c = f * c + i * torch.tanh(z)
                n = f * n + i
                h = torch.sigmoid(o) * c / n
                m = m_new
                outputs.append(h)
            x = torch.stack(outputs, dim=1)
        return x if self.return_sequences else x[:, -1]
'''

AUDIO_FRONTEND = '''\
import math


def mel_filterbank(n_fft: int, n_mels: int, sample_rate: int, f_min: float = 0.0,
                   f_max: float | None = None) -> torch.Tensor:
    """HTK mel filterbank [n_mels, n_fft // 2 + 1] (triangular, unnormalised)."""
    f_max = f_max or sample_rate / 2
    hz_to_mel = lambda f: 2595.0 * math.log10(1.0 + f / 700.0)  # noqa: E731
    mels = torch.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    hz = 700.0 * (10 ** (mels / 2595.0) - 1.0)
    bins = torch.linspace(0, sample_rate / 2, n_fft // 2 + 1)
    lower, center, upper = hz[:-2, None], hz[1:-1, None], hz[2:, None]
    up = (bins - lower) / (center - lower)
    down = (upper - bins) / (upper - center)
    return torch.clamp(torch.minimum(up, down), min=0.0)


class MelSpectrogram(nn.Module):
    """Waveform [B, 1, samples] -> (log) mel spectrogram [B, 1, n_mels, frames] on any device
    ([B, frames, n_mels] with sequence=True)."""

    def __init__(self, sample_rate: int = 16000, n_fft: int = 400, hop_length: int = 160,
                 n_mels: int = 64, log: bool = True, n_mfcc: int = 0,
                 sequence: bool = False) -> None:
        super().__init__()
        self.n_fft, self.hop, self.log, self.n_mfcc = n_fft, hop_length, log, n_mfcc
        self.sequence = sequence
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.register_buffer("fb", mel_filterbank(n_fft, n_mels, sample_rate), persistent=False)
        if n_mfcc:
            k = torch.arange(n_mels).float()
            dct = torch.cos(math.pi / n_mels * (k[None, :] + 0.5)
                            * torch.arange(n_mfcc).float()[:, None])
            self.register_buffer("dct", dct, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wave = x.reshape(x.shape[0], -1)
        spec = torch.stft(wave, self.n_fft, self.hop, window=self.window, center=True,
                          return_complex=True).abs().pow(2)          # [B, F, T]
        mel = torch.matmul(self.fb, spec)                             # [B, M, T]
        if self.log or self.n_mfcc:
            mel = torch.log(mel + 1e-6)
        if self.n_mfcc:
            mel = torch.matmul(self.dct, mel)
        return mel.transpose(1, 2) if self.sequence else mel.unsqueeze(1)
'''

HF_AUDIO = '''\
import math


class HFAudioEncoder(nn.Module):
    """Hugging Face speech encoder (wav2vec 2.0, HuBERT, WavLM, Whisper encoder).

    Waveform [B, 1, samples] at 16 kHz -> pooled [B, F] or frame features [B, T, F].
    """

    PRESETS = {  # repo: (model class, config class, config kwargs for random init)
        "facebook/wav2vec2-base": ("Wav2Vec2Model", "Wav2Vec2Config", {}),
        "facebook/hubert-base-ls960": ("HubertModel", "HubertConfig", {}),
        "microsoft/wavlm-base-plus": ("WavLMModel", "WavLMConfig", {}),
        "openai/whisper-tiny": ("WhisperModel", "WhisperConfig",
                                {"d_model": 384, "encoder_layers": 4, "decoder_layers": 4,
                                 "encoder_attention_heads": 6, "decoder_attention_heads": 6,
                                 "encoder_ffn_dim": 1536, "decoder_ffn_dim": 1536}),
        "openai/whisper-base": ("WhisperModel", "WhisperConfig",
                                {"d_model": 512, "encoder_layers": 6, "decoder_layers": 6,
                                 "encoder_attention_heads": 8, "decoder_attention_heads": 8,
                                 "encoder_ffn_dim": 2048, "decoder_ffn_dim": 2048}),
    }

    def __init__(self, model_id: str, pretrained: bool = False, freeze: bool = True,
                 pooling: str = "mean") -> None:
        super().__init__()
        import transformers

        model_cls, config_cls, kwargs = self.PRESETS[model_id]
        model_cls = getattr(transformers, model_cls)
        self.whisper = "whisper" in model_id
        self.pooling = pooling
        if pretrained and not skip_pretrained():
            model = model_cls.from_pretrained(model_id)
        else:
            model = model_cls(getattr(transformers, config_cls)(**kwargs))
        self.model = model.get_encoder() if self.whisper else model
        if self.whisper:
            self.mel = MelSpectrogram(16000, 400, 160, self.model.config.num_mel_bins, log=False)
        if freeze and pretrained:
            for p in self.model.parameters():
                p.requires_grad = False

    def frames(self, x: torch.Tensor) -> torch.Tensor:
        wave = x.reshape(x.shape[0], -1)
        if not self.whisper:
            wave = (wave - wave.mean(-1, keepdim=True)) / (wave.std(-1, keepdim=True) + 1e-7)
            return self.model(input_values=wave).last_hidden_state
        n = wave.shape[-1]
        wave = nn.functional.pad(wave, (0, max(0, 480000 - n)))[:, :480000]  # 30 s window
        mel = self.mel(wave)[:, 0, :, :3000].clamp(min=1e-10)
        logmel = torch.log10(mel)  # Whisper's log-mel scaling
        logmel = torch.maximum(logmel, logmel.amax(dim=(1, 2), keepdim=True) - 8.0)
        hidden = self.model(input_features=(logmel + 4.0) / 4.0).last_hidden_state
        return hidden[:, : max(1, math.ceil(n / 320))]                # frames with audio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.frames(x)
        return hidden.mean(1) if self.pooling == "mean" else hidden
'''

SEQUENCE_HELPERS: dict[str, str] = {
    "TCN": TCN, "MambaBlock": MAMBA, "S4DLayer": S4D, "SLSTM": SLSTM,
    "MelSpectrogram": AUDIO_FRONTEND, "HFAudioEncoder": HF_AUDIO,
}


def register() -> None:
    from ai_made_easy.core.codegen.helpers import PYTORCH_HELPERS

    PYTORCH_HELPERS.update(SEQUENCE_HELPERS)


# ------------------------------------------------------------- analytic costs
# (parameters, multiply-accumulates per sample)

def tcn_cost(cin: int, channels: int, levels: int, kernel: int, length: int) -> tuple[int, int]:
    params = macs = 0
    for i in range(levels):
        a = cin if i == 0 else channels
        conv = a * channels * kernel + channels + channels * channels * kernel + channels
        params += conv
        macs += (a * channels * kernel + channels * channels * kernel) * length
        if a != channels:
            params += a * channels + channels
            macs += a * channels * length
    return params, macs


def tcn_receptive_field(levels: int, kernel: int) -> int:
    return 1 + 2 * (kernel - 1) * (2 ** levels - 1)


def mamba_cost(dim: int, d_state: int, d_conv: int, expand: int,
               length: int) -> tuple[int, int]:
    inner = expand * dim
    rank = max(1, math.ceil(dim / 16))
    params = (2 * dim                                   # layer norm
              + dim * 2 * inner                         # in_proj
              + inner * d_conv + inner                  # depthwise conv
              + inner * (rank + 2 * d_state)            # x_proj
              + rank * inner + inner                    # dt_proj
              + inner * d_state + inner                 # A_log, D
              + inner * dim)                            # out_proj
    macs = length * (dim * 2 * inner + inner * d_conv + inner * (rank + 2 * d_state)
                     + rank * inner + 3 * inner * d_state + inner * dim)
    return params, macs


def s4d_cost(dim: int, d_state: int, length: int) -> tuple[int, int]:
    half = d_state // 2
    params = 2 * dim + dim + 2 * dim * half + 2 * dim * half + dim + dim * dim + dim
    fft = int(2 * length * math.log2(max(2 * length, 2)))
    macs = length * (dim * dim + dim) + dim * half * length + 3 * dim * fft
    return params, macs


def slstm_cost(cin: int, hidden: int, layers: int, length: int) -> tuple[int, int]:
    params = macs = 0
    for i in range(layers):
        a = cin if i == 0 else hidden
        params += a * 4 * hidden + 4 * hidden + hidden * 4 * hidden
        macs += length * (a * 4 * hidden + hidden * 4 * hidden + 8 * hidden)
    return params, macs


def frames(samples: int, hop: int) -> int:
    return samples // hop + 1  # torch.stft(center=True)


def wav2vec_frames(samples: int) -> int:
    """Frames after the wav2vec 2.0 / HuBERT / WavLM convolutional feature encoder."""
    length = samples
    for kernel, stride in ((10, 5), (3, 2), (3, 2), (3, 2), (3, 2), (2, 2), (2, 2)):
        length = (length - kernel) // stride + 1
    return length
