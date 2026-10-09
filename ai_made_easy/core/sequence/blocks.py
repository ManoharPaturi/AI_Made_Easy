"""Sequence layers (TCN, Mamba, S4D, xLSTM) and in-model audio front ends / encoders."""
from __future__ import annotations

from dataclasses import replace

from ai_made_easy.core.blocks._dsl import P, ShapeError, nn_block
from ai_made_easy.core.registry import get_registry
from ai_made_easy.core.sequence import helpers as h

SEQUENCE = "Sequence Models"
AUDIO = "Audio Front-End"
HF_AUDIO = {  # model id: (feature width, frames function, parameters in M, GMACs per second)
    "facebook/wav2vec2-base": (768, h.wav2vec_frames, 94.4, 17.6),
    "facebook/hubert-base-ls960": (768, h.wav2vec_frames, 94.4, 17.6),
    "microsoft/wavlm-base-plus": (768, h.wav2vec_frames, 94.4, 17.9),
    "openai/whisper-tiny": (384, lambda s: max(1, -(-s // 320)), 8.2, 4.0),
    "openai/whisper-base": (512, lambda s: max(1, -(-s // 320)), 20.6, 9.5),
}


def _sequence(shape: list[int], name: str) -> None:
    if len(shape) != 2:
        raise ShapeError(f"{name} expects a batch-first sequence [L, C], got {shape}")


def _with_cost(defn, cost, **changes):
    """Attach meta['cost'] = (params, FLOPs) for the summary and budgets."""
    return replace(defn, meta={"cost": cost}, **changes)


# ------------------------------------------------------------------ sequence layers

def _tcn_shape(in_shapes, p):
    _sequence(in_shapes[0], "TCN")
    return [in_shapes[0][0], int(p["channels"])]


def _tcn_cost(in_shapes, p):
    s = in_shapes[0]
    params, macs = h.tcn_cost(s[1], int(p["channels"]), int(p["levels"]),
                              int(p["kernel_size"]), s[0])
    return params, 2 * macs


def _same_shape(name: str):
    def fn(in_shapes, p):
        _sequence(in_shapes[0], name)
        return list(in_shapes[0])
    return fn


def _mamba_cost(in_shapes, p):
    s = in_shapes[0]
    params, macs = h.mamba_cost(s[1], int(p["d_state"]), int(p["d_conv"]), int(p["expand"]),
                                s[0])
    return params, 2 * macs


def _s4d_cost(in_shapes, p):
    s = in_shapes[0]
    params, macs = h.s4d_cost(s[1], int(p["d_state"]), s[0])
    return params, 2 * macs


def _s4d_checks(p):
    if int(p["d_state"]) % 2:
        return [("error", "d_state must be even (complex-conjugate state pairs)")]
    return []


def _xlstm_shape(in_shapes, p):
    _sequence(in_shapes[0], "xLSTM")
    width = int(p["hidden_size"])
    return [in_shapes[0][0], width] if p["return_sequences"] else [width]


def _xlstm_cost(in_shapes, p):
    s = in_shapes[0]
    params, macs = h.slstm_cost(s[1], int(p["hidden_size"]), int(p["num_layers"]), s[0])
    return params, 2 * macs


# ------------------------------------------------------------------ audio

def _samples(shape: list[int], name: str) -> int:
    if len(shape) == 1 or (len(shape) == 2 and shape[0] == 1):
        return shape[-1]
    raise ShapeError(f"{name} expects a mono waveform [1, samples] or [samples], got {shape}")


def _mel_shape(kind: str):
    def fn(in_shapes, p):
        samples = _samples(in_shapes[0], kind)
        n_fft = int(p["n_fft"])
        if samples <= n_fft // 2:
            raise ShapeError(f"{kind} needs more than {n_fft // 2} samples (n_fft = {n_fft}), "
                             f"got {samples}")
        if int(p["hop_length"]) > n_fft:
            raise ShapeError("hop_length larger than n_fft skips audio between frames")
        bins = int(p["n_mfcc"]) if kind == "MFCC" else int(p["n_mels"])
        frames = h.frames(samples, int(p["hop_length"]))
        return [frames, bins] if str(p.get("layout", "")).startswith("sequence") \
            else [1, bins, frames]
    return fn


LAYOUT = P("layout", "enum", "image [1, bins, frames]",
           options=("image [1, bins, frames]", "sequence [frames, bins]"),
           help="image for Conv2D; sequence for RNNs, Transformers and CTC speech recognition")


def _mel_checks(p):
    out = []
    if int(p["n_mels"]) > int(p["n_fft"]) // 2 + 1:
        out.append(("warning", "more mel bins than frequency bins: some filters are empty; "
                               "lower n_mels or raise n_fft"))
    if "n_mfcc" in p and int(p["n_mfcc"]) > int(p["n_mels"]):
        out.append(("error", "n_mfcc cannot exceed n_mels"))
    return out


def _mel_cost(in_shapes, p):
    samples = _samples(in_shapes[0], "audio")
    n_fft, hop = int(p["n_fft"]), int(p["hop_length"])
    frames = h.frames(samples, hop)
    fft = int(5 * n_fft * max(1, n_fft.bit_length()))
    macs = frames * (fft // 2 + int(p["n_mels"]) * (n_fft // 2 + 1)
                     + int(p.get("n_mfcc", 0)) * int(p["n_mels"]))
    return 0, 2 * macs


def _mel_torch(kind: str):
    def fn(c):
        mfcc = f", n_mfcc={int(c['n_mfcc'])}" if kind == "mfcc" else ""
        seq = ", sequence=True" if str(c.get("layout", "")).startswith("sequence") else ""
        return (f"MelSpectrogram(sample_rate={int(c['sample_rate'])}, n_fft={int(c['n_fft'])}, "
                f"hop_length={int(c['hop_length'])}, n_mels={int(c['n_mels'])}, "
                f"log={bool(c.get('log', True))}{mfcc}{seq})")
    return fn


def _hf_audio_shape(in_shapes, p):
    samples = _samples(in_shapes[0], "Speech encoder")
    width, frames, _m, _g = HF_AUDIO[p["model_id"]]
    if samples < 400:
        raise ShapeError("speech encoders need at least 400 samples (25 ms at 16 kHz)")
    if "whisper" in p["model_id"] and samples > 480000:
        raise ShapeError("Whisper encodes at most 30 s (480,000 samples at 16 kHz)")
    t = frames(samples)
    return [width] if p["pooling"] == "mean" else [t, width]


def _hf_audio_cost(in_shapes, p):
    samples = _samples(in_shapes[0], "audio")
    _w, _f, params_m, gmacs = HF_AUDIO[p["model_id"]]
    seconds = 30.0 if "whisper" in p["model_id"] else samples / 16000
    return int(params_m * 1e6), int(2 * gmacs * 1e9 * seconds)


def _blocks() -> list:
    tcn = nn_block(
        "seq.tcn", "Temporal Conv Net (TCN)", SEQUENCE, family="recurrent",
        params=(P("channels", "int", 64, lo=1), P("levels", "int", 4, lo=1, hi=12,
                                                    help="Blocks with dilation 1, 2, 4, ..."),
                P("kernel_size", "int", 3, lo=2), P("dropout", "float", 0.1, lo=0.0, hi=1.0)),
        shape=_tcn_shape, layout="ir",
        param_fn=lambda s, p: _tcn_cost(s, p)[0],
        torch=lambda c: (f"TCN({c['input_size']}, {int(c['channels'])}, {int(c['levels'])}, "
                         f"kernel={int(c['kernel_size'])}, dropout={float(c['dropout'])})"),
        torch_helpers=("TCN",),
        desc="Stack of dilated causal convolutions with residuals: [L, C] → [L, channels]. "
             "Each output step sees only the past.")
    mamba = nn_block(
        "seq.mamba", "Mamba (Selective SSM)", SEQUENCE, family="recurrent",
        params=(P("d_state", "int", 16, lo=1, help="State size per channel"),
                P("d_conv", "int", 4, lo=1, help="Causal convolution width"),
                P("expand", "int", 2, lo=1, help="Inner width = expand × channels")),
        shape=_same_shape("Mamba"), layout="ir", param_fn=lambda s, p: _mamba_cost(s, p)[0],
        torch=lambda c: (f"MambaBlock({c['input_size']}, d_state={int(c['d_state'])}, "
                         f"d_conv={int(c['d_conv'])}, expand={int(c['expand'])})"),
        torch_helpers=("MambaBlock",),
        desc="Selective state-space block with input-dependent dynamics (Mamba), residual: "
             "[L, C] → [L, C]. Linear in sequence length.")
    s4d = nn_block(
        "seq.s4d", "S4D (Structured SSM)", SEQUENCE, family="recurrent",
        params=(P("d_state", "int", 64, lo=2, help="State size (even)"),
                P("dropout", "float", 0.0, lo=0.0, hi=1.0)),
        shape=_same_shape("S4D"), layout="ir", checks=_s4d_checks,
        param_fn=lambda s, p: _s4d_cost(s, p)[0],
        torch=lambda c: (f"S4DLayer({c['input_size']}, d_state={int(c['d_state'])}, "
                         f"dropout={float(c['dropout'])})"),
        torch_helpers=("S4DLayer",),
        desc="Diagonal state-space layer applied as a long FFT convolution, residual: "
             "[L, C] → [L, C]. Captures very long dependencies.")
    xlstm = nn_block(
        "seq.xlstm", "xLSTM (sLSTM)", SEQUENCE, family="recurrent",
        params=(P("hidden_size", "int", 64, lo=1), P("num_layers", "int", 1, lo=1),
                P("return_sequences", "bool", True,
                  help="True: every time step [L, H]; False: final state [H]")),
        shape=_xlstm_shape, layout="ir", param_fn=lambda s, p: _xlstm_cost(s, p)[0],
        torch=lambda c: (f"SLSTM({c['input_size']}, {int(c['hidden_size'])}, "
                         f"layers={int(c['num_layers'])}, "
                         f"return_sequences={bool(c['return_sequences'])})"),
        torch_helpers=("SLSTM",),
        desc="sLSTM from xLSTM: exponential gating with a normaliser state; "
             "[L, C] → [L, H] or [H].")
    mel_params = (P("sample_rate", "int", 16000, lo=1000), P("n_fft", "int", 400, lo=16),
                  P("hop_length", "int", 160, lo=1), P("n_mels", "int", 64, lo=1))
    mel = nn_block(
        "audio.mel_spectrogram", "Mel Spectrogram", AUDIO, family="tensor",
        params=(*mel_params, P("log", "bool", True, help="Log-compress the energies"), LAYOUT),
        shape=_mel_shape("Mel Spectrogram"), checks=_mel_checks,
        torch=_mel_torch("mel"), torch_helpers=("MelSpectrogram",),
        desc="Waveform [1, samples] → mel spectrogram [1, n_mels, frames], computed inside "
             "the model (runs on the GPU, exported with it). Sequence layout gives "
             "[frames, n_mels] for RNNs / Transformers / CTC.")
    mfcc = nn_block(
        "audio.mfcc", "MFCC", AUDIO, family="tensor",
        params=(*mel_params, P("n_mfcc", "int", 13, lo=1), LAYOUT),
        shape=_mel_shape("MFCC"), checks=_mel_checks,
        torch=_mel_torch("mfcc"), torch_helpers=("MelSpectrogram",),
        desc="Waveform [1, samples] → MFCCs [1, n_mfcc, frames] (or [frames, n_mfcc]) inside "
             "the model.")
    encoder = nn_block(
        "audio.hf_encoder", "Speech Encoder (wav2vec 2.0 / HuBERT / Whisper)",
        "Pretrained Models", family="model",
        params=(P("model_id", "enum", "facebook/wav2vec2-base", options=tuple(HF_AUDIO)),
                P("weights", "enum", "pretrained", options=("pretrained", "none")),
                P("freeze", "bool", True),
                P("pooling", "enum", "mean", options=("mean", "none"),
                  help="mean → [F]; none → frame features [T, F] (for CTC speech "
                       "recognition)")),
        shape=_hf_audio_shape,
        param_fn=lambda s, p: _hf_audio_cost(s, p)[0],
        torch=lambda c: (f"HFAudioEncoder(\"{c['model_id']}\", "
                         f"pretrained={c['weights'] == 'pretrained'}, "
                         f"freeze={bool(c['freeze'])}, pooling=\"{c['pooling']}\")"),
        torch_helpers=("skip_pretrained", "MelSpectrogram", "HFAudioEncoder"),
        desc="Self-supervised speech model over 16 kHz waveforms: utterance embedding [F] "
             "or frame features [T, F].")
    return [
        _with_cost(tcn, _tcn_cost),
        _with_cost(mamba, _mamba_cost),
        _with_cost(s4d, _s4d_cost),
        _with_cost(xlstm, _xlstm_cost),
        _with_cost(mel, _mel_cost),
        _with_cost(mfcc, _mel_cost),
        _with_cost(encoder, _hf_audio_cost, requires=("transformers",), extra="audio"),
    ]


def register_all() -> None:
    reg = get_registry()
    for defn in _blocks():
        reg.register(defn)
