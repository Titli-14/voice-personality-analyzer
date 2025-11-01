from flask import Flask, render_template, request, jsonify
import librosa
import numpy as np
import os
import uuid

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def clamp01(x):
    try:
        return max(0.0, min(1.0, float(np.mean(x))))
    except Exception:
        return 0.0

def scale_to_0_100(x):
    return int(round(clamp01(x) * 100))

def analyze_voice(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        if y.size == 0:
            return {"error": "Empty audio"}

        y, _ = librosa.effects.trim(y)
        duration = max(0.001, librosa.get_duration(y=y, sr=sr))
        rms = np.mean(librosa.feature.rms(y=y))
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))

        try:
            f0 = librosa.yin(y, fmin=50, fmax=400, sr=sr)
            f0 = f0[~np.isnan(f0)]
            avg_pitch = float(np.mean(f0)) if f0.size > 0 else 0.0
            pitch_std = float(np.std(f0)) if f0.size > 0 else 0.0
        except Exception:
            avg_pitch = 0.0
            pitch_std = 0.0

        spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_var = float(np.mean(np.std(mfcc, axis=1)))
        rms_frames = librosa.feature.rms(y=y)[0]
        silence_thresh = np.percentile(rms_frames, 25) * 0.5
        pause_ratio = float(np.sum(rms_frames < silence_thresh) / max(1, rms_frames.size))

        energy_norm = clamp01((rms - 0.001) / 0.08)
        conf_raw = 0.6 * energy_norm + 0.3 * (1 - pause_ratio) + 0.1 * (1 - clamp01(zcr * 5))
        confidence = scale_to_0_100(conf_raw)
        calm_raw = 0.5 * (1 - energy_norm) + 0.3 * (1 - clamp01(pitch_std / 50)) + 0.2 * (1 - clamp01(tempo / 140))
        calmness = scale_to_0_100(calm_raw)
        energy = scale_to_0_100(energy_norm)
        warmth_raw = 0.5 * (1 - clamp01(spec_cent / 3000)) + 0.5 * (1 - clamp01(mfcc_var / 20))
        warmth = scale_to_0_100(warmth_raw)
        speed_norm = clamp01((tempo - 60) / 120)
        speaking_speed = scale_to_0_100(speed_norm)
        pitch_var_norm = clamp01(pitch_std / 60)
        pitch_variability = scale_to_0_100(pitch_var_norm)
        charm_raw = 0.45 * conf_raw + 0.35 * pitch_var_norm + 0.2 * energy_norm
        charm = scale_to_0_100(charm_raw)

        interpretations = {
            "confidence": (
                "Speaks with conviction and presence."
                if confidence > 65 else
                "Somewhat hesitant or soft-spoken." if confidence < 40 else
                "Moderately confident."
            ),
            "calmness": (
                "Very calm and steady." if calmness > 65 else
                "A bit tense or energetic." if calmness < 40 else
                "Relatively balanced."
            ),
            "warmth": (
                "Warm, soothing timbre." if warmth > 60 else
                "Neutral or bright tone." if warmth < 40 else
                "Pleasant tone."
            ),
            "speaking_speed": (
                "Fast speaker" if speaking_speed > 70 else
                "Slow speaker" if speaking_speed < 30 else
                "Moderate pace"
            ),
            "pitch_variability": (
                "Expressive voice" if pitch_variability > 65 else
                "Monotone" if pitch_variability < 30 else
                "Some variation"
            ),
            "charm": (
                "Outgoing & lively" if charm > 65 else
                "Reserved" if charm < 35 else
                "Friendly"
            )
        }

        return {
            "duration_seconds": round(float(duration), 2),
            "avg_pitch_hz": round(float(avg_pitch), 1),
            "pitch_std_hz": round(float(pitch_std), 1),
            "rms": float(rms),
            "tempo_bpm": round(float(tempo), 1),
            "pause_ratio": round(float(pause_ratio), 3),
            "scores": {
                "confidence": confidence,
                "energy": energy,
                "calmness": calmness,
                "warmth": warmth,
                "speaking_speed": speaking_speed,
                "pitch_variability": pitch_variability,
                "charm": charm
            },
            "interpretations": interpretations
        }

    except Exception as e:
        print("❌ Error analyzing voice:", e)
        return {"error": str(e)}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    file = request.files['audio']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = f"{uuid.uuid4().hex}.wav"
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)
    print(f"✅ Saved temporary audio file: {path}")

    result = analyze_voice(path)

    # 🧹 Auto delete the file after analysis
    try:
        os.remove(path)
        print(f"🗑️ Deleted: {path}")
    except Exception as e:
        print(f"⚠️ Couldn't delete file: {e}")

    print("🎯 Analysis result:", result)
    return jsonify(result)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

