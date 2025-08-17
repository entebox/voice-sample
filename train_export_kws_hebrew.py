#!/usr/bin/env python3
# train_export_kws_hebrew.py
#
# Trains a tiny keyword-spotting model that ingests RAW 1s waveform @ 16kHz
# and outputs 2 scores: [non_keyword, keyword].
#
# Matches your KWSGate expectations:
#   - Input:  [1, 16000] float32 (raw PCM normalized inside the graph)
#   - Output: [1, 2] float32 softmax (index 1 == "keyword")
#
# Usage:
#   python3 train_export_kws_hebrew.py \
#       --data_dir ./data_kws \
#       --epochs 25 \
#       --batch_size 64 \
#       --out kws_hebrew.tflite
#
# After training, copy the .tflite to the device and mount it to /app/models/kws_hebrew.tflite

import argparse
import os
import random
from glob import glob

import numpy as np
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Folder with subdirs 'keyword' and 'background'")
    ap.add_argument("--sr", type=int, default=16000, help="Target sample rate (must be 16000 for your runtime)")
    ap.add_argument("--duration", type=float, default=1.0, help="Clip length (seconds)")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--val_split", type=float, default=0.15)
    ap.add_argument("--test_split", type=float, default=0.10)
    ap.add_argument("--out", default="kws_hebrew.tflite")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

def list_wavs(dirpath):
    return sorted(glob(os.path.join(dirpath, "*.wav")) + glob(os.path.join(dirpath, "*.WAV")))

def split_files(files, val_split, test_split, seed=42):
    rnd = random.Random(seed)
    files = files[:]
    rnd.shuffle(files)
    n = len(files)
    n_val = int(n * val_split)
    n_test = int(n * test_split)
    val = files[:n_val]
    test = files[n_val:n_val+n_test]
    train = files[n_val+n_test:]
    return train, val, test

def decode_and_resample(path, target_sr):
    audio = tf.io.read_file(path)
    wav, sr = tf.audio.decode_wav(audio, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)  # [samples]

    # Resample if needed
    sr = tf.cast(sr, tf.int32)
    def _resamp(x):
        # tf.signal.resample is not stable across TF versions; use tfio if available
        # Fallback: linear interpolation via tf.signal.resample_poly (TF>=2.12) or simple tf.signal.stft ratio adjust.
        # Simpler route: if sr != target_sr, use tf.signal.resample.
        ratio = tf.cast(target_sr, tf.float32) / tf.cast(sr, tf.float32)
        new_len = tf.cast(tf.math.round(tf.cast(tf.shape(x)[0], tf.float32) * ratio), tf.int32)
        return tf.image.resize(tf.expand_dims(x, 0), [1, new_len], method="bilinear")[0,0,:]

    wav = tf.cond(tf.not_equal(sr, target_sr), lambda: _resamp(wav), lambda: wav)
    return wav

def random_crop_or_pad_1s(wav, sr, duration=1.0, seed=None):
    needed = int(sr * duration)
    n = tf.shape(wav)[0]
    # Random shift / crop
    def crop():
        start = tf.random.uniform([], 0, n - needed + 1, dtype=tf.int32, seed=seed)
        return wav[start:start+needed]
    def pad():
        pad_amt = needed - n
        pad_left = tf.random.uniform([], 0, pad_amt + 1, dtype=tf.int32, seed=seed)
        pad_right = pad_amt - pad_left
        return tf.pad(wav, [[pad_left, pad_right]])

    return tf.cond(n > needed, crop, pad)

def augment(wav, sr):
    # Simple augmentations: gain jitter and small time shift
    gain = tf.random.uniform([], 0.8, 1.2)
    wav = tf.clip_by_value(wav * gain, -1.0, 1.0)

    # tiny time shift by up to +/- 60 ms
    max_shift = int(0.06 * sr)
    shift = tf.random.uniform([], -max_shift, max_shift+1, dtype=tf.int32)
    wav = tf.roll(wav, shift=shift, axis=0)
    return wav

# Feature extractor (inside the graph): log-mel from raw
def wav_to_logmel(wav, sr, n_mels=40, frame_len=400, frame_step=160, fft_len=512, fmin=20.0, fmax=4000.0):
    # Pre-emphasis (light)
    wav = tf.cast(wav, tf.float32)
    wav = tf.clip_by_value(wav, -1.0, 1.0)

    stft = tf.signal.stft(wav, frame_length=frame_len, frame_step=frame_step, fft_length=fft_len, window_fn=tf.signal.hann_window)
    mag = tf.abs(stft)  # [T, 1+fft_len/2]

    num_spec_bins = mag.shape[-1]
    mel_w = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=n_mels,
        num_spectrogram_bins=num_spec_bins,
        sample_rate=sr,
        lower_edge_hertz=fmin,
        upper_edge_hertz=fmax,
    )
    mel = tf.tensordot(tf.square(mag), mel_w, axes=1)  # [T, M]
    mel = tf.math.log(mel + 1e-6)
    return mel  # [time, mels]

def make_model(sr=16000, duration=1.0):
    inp = tf.keras.Input(shape=(int(sr*duration),), name="waveform")

    # Log-mel features in-graph
    x = tf.keras.layers.Lambda(lambda w: wav_to_logmel(w, sr))(inp)              # [T, M]
    x = tf.keras.layers.LayerNormalization(axis=[1, 2])(tf.keras.layers.Reshape((-1, 40, 1))(x))

    # Tiny CNN
    x = tf.keras.layers.SeparableConv2D(16, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.SeparableConv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.SeparableConv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    out = tf.keras.layers.Dense(2, activation="softmax", name="probs")(x)  # [non, keyword]

    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def build_dataset(data_dir, sr, duration, batch_size, val_split, test_split, seed=42):
    kw_files = list_wavs(os.path.join(data_dir, "keyword"))
    bg_files = list_wavs(os.path.join(data_dir, "background"))
    if not kw_files or not bg_files:
        raise RuntimeError("Expected subfolders 'keyword' and 'background' with wav files.")

    # splits
    kw_tr, kw_val, kw_te = split_files(kw_files, val_split, test_split, seed=seed)
    bg_tr, bg_val, bg_te = split_files(bg_files, val_split, test_split, seed=seed)

    def make_ds(files, label, training):
        lbl = tf.convert_to_tensor(label, dtype=tf.int32)
        ds = tf.data.Dataset.from_tensor_slices(files)

        def _load(path):
            wav = decode_and_resample(path, sr)
            wav = random_crop_or_pad_1s(wav, sr, duration)
            if training:
                wav = augment(wav, sr)
            return wav, lbl

        ds = ds.shuffle(2048, seed=seed) if training else ds
        ds = ds.map(lambda p: _load(p), num_parallel_calls=AUTOTUNE)
        return ds

    ds_tr = make_ds(kw_tr, 1, True).concatenate(make_ds(bg_tr, 0, True)).shuffle(4096, seed=seed).batch(batch_size).prefetch(AUTOTUNE)
    ds_val = make_ds(kw_val, 1, False).concatenate(make_ds(bg_val, 0, False)).batch(batch_size).prefetch(AUTOTUNE)
    ds_te  = make_ds(kw_te,  1, False).concatenate(make_ds(bg_te,  0, False)).batch(batch_size).prefetch(AUTOTUNE)
    return ds_tr, ds_val, ds_te

def export_tflite(model, out_path):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    # Keep it float32 to avoid tricky full-int8 constraints with the in-graph STFT/mel.
    # (The model is tiny; CPU is fine. If you later want int8, move feature extraction outside the graph.)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = conv.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    print(f"Saved TFLite to: {out_path}")

def quick_tflite_check(out_path):
    import tflite_runtime.interpreter as tflite
    interp = tflite.Interpreter(model_path=out_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    print("TFLite input:", inp["shape"], inp["dtype"])
    print("TFLite output:", out["shape"], out["dtype"])

    # random test
    sr = 16000
    x = np.zeros((1, sr), dtype=np.float32)
    interp.set_tensor(inp["index"], x)
    interp.invoke()
    y = interp.get_tensor(out["index"])
    print("Dummy inference OK, output:", y)

def main():
    args = parse_args()
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    ds_tr, ds_val, ds_te = build_dataset(args.data_dir, args.sr, args.duration,
                                         args.batch_size, args.val_split, args.test_split, seed=args.seed)

    model = make_model(sr=args.sr, duration=args.duration)
    model.summary()

    ckpt = tf.keras.callbacks.ModelCheckpoint("kws_best.keras", monitor="val_accuracy",
                                              save_best_only=True, save_weights_only=False, mode="max")
    es = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, mode="max", restore_best_weights=True)

    model.fit(ds_tr, validation_data=ds_val, epochs=args.epochs, callbacks=[ckpt, es])

    print("Evaluating on test set…")
    test_loss, test_acc = model.evaluate(ds_te)
    print(f"Test acc: {test_acc:.3f}")

    export_tflite(model, args.out)
    quick_tflite_check(args.out)

if __name__ == "__main__":
    main()