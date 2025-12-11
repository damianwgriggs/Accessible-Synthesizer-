import numpy as np
import scipy.io.wavfile as wavfile
import requests
import secrets
import hashlib
import time
import json
import google.generativeai as genai

# --- CONFIGURATION ---
# !!! PASTE YOUR API KEY HERE !!!
GOOGLE_API_KEY = "PASTEKEYHERE"

MODEL_NAME = 'gemini-2.5-flash' 
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

SAMPLE_RATE = 44100

# --- 1. THE CHAOS ENGINE ---
def get_quantum_seed():
    hw = secrets.token_bytes(32)
    qw = b''
    try:
        url = "https://qrng.anu.edu.au/API/jsonI.php?length=1&type=hex16&size=32"
        r = requests.get(url, timeout=1)
        if r.status_code == 200:
            qw = bytes.fromhex(r.json()['data'][0])
    except:
        pass
    tw = str(time.time_ns()).encode()
    hasher = hashlib.sha256()
    hasher.update(hw + qw + tw)
    seed = int(hasher.hexdigest(), 16) % (2**32)
    print(f"--- QUANTUM SEED: {seed} ---")
    return seed

# --- 2. THE COMPOSER (A/B Structure) ---
def compose_dual_sections(seed):
    print(f"Contacting {MODEL_NAME} for A/B Song Structure...")
    
    prompt = f"""
    You are a Neo-Soul/Hip-Hop producer.
    Seed: {seed}.
    
    Create a song structure with TWO distinct sections (A and B).
    
    JSON Structure:
    {{
      "bpm": integer (80-95),
      "swing": float (0.05-0.12),
      "section_a": {{
        "kick": [list of steps 0-63],
        "snare": [list of steps],
        "closed_hat": [list of steps],
        "open_hat": [list of steps],
        "bass": [ {{ "step": int, "freq": float, "dur": int }} ],
        "keys": [ {{ "step": int, "chord_freqs": [floats], "dur": int }} ]
      }},
      "section_b": {{
        "kick": [list of steps 0-63],
        "snare": [list of steps],
        "closed_hat": [list of steps],
        "open_hat": [list of steps],
        "bass": [ {{ "step": int, "freq": float, "dur": int }} ],
        "keys": [ {{ "step": int, "chord_freqs": [floats], "dur": int }} ]
      }}
    }}
    
    Creative Rules:
    1. **Section A (Verse)**: Chill, sparse, laid back.
    2. **Section B (Spice/Bridge)**: Same Key/BPM, but spice it up! Add more syncopation to the drums, change the chord progression slightly, or make the bassline busier.
    3. Output strictly JSON.
    """
    
    try:
        response = model.generate_content(prompt)
        text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        print(f"Gemini Error: {e}")
        return None # Will handle in main

# --- 3. DSP SYNTHESIS (Smooth) ---

def mix(master, sound, loc, vol=1.0):
    if loc < 0:
        offset = abs(loc)
        if offset >= len(sound): return
        sound = sound[offset:]
        loc = 0
    if loc + len(sound) >= len(master):
        avail = len(master) - loc
        if avail <= 0: return
        sound = sound[:avail]
    master[loc:loc+len(sound)] += sound * vol

def synth_kick_smooth():
    dur = 0.45
    t = np.linspace(0, dur, int(SAMPLE_RATE * dur))
    freq = 100 * np.exp(-10 * t) + 40 
    sub = np.sin(2 * np.pi * freq * t)
    env = np.exp(-4 * t)
    noise = np.random.uniform(-1, 1, len(t)) * 0.05
    return (sub * env + noise) * 0.9

def synth_snare_smooth():
    dur = 0.3
    t = np.linspace(0, dur, int(SAMPLE_RATE * dur))
    tone = np.sin(2 * np.pi * 160 * t) * np.exp(-15 * t)
    noise = np.random.uniform(-1, 1, len(t)) * np.exp(-12 * t)
    return (tone * 0.6 + noise * 0.3) * 0.8

def synth_hat_smooth(open_h=False):
    dur = 0.4 if open_h else 0.08
    t = np.linspace(0, dur, int(SAMPLE_RATE * dur))
    noise = np.random.uniform(-1, 1, len(t))
    mod = np.sin(2 * np.pi * 12000 * t) 
    env = np.exp(-6 * t) if open_h else np.exp(-35 * t)
    return noise * mod * env * 0.25

def synth_bass_warm(freq, steps, step_len):
    dur = steps * step_len
    t = np.linspace(0, dur, int(SAMPLE_RATE * dur))
    wave = np.sin(2 * np.pi * freq * t) + np.sin(2 * np.pi * (freq * 2) * t) * 0.2
    env = np.ones_like(t)
    attack = int(0.05 * SAMPLE_RATE)
    env[:attack] = np.linspace(0, 1, attack)
    env[-2000:] = np.linspace(1, 0, 2000)
    return wave * env * 0.7

def synth_keys_lush(freqs, steps, step_len):
    dur = steps * step_len
    t = np.linspace(0, dur, int(SAMPLE_RATE * dur))
    master_chord = np.zeros_like(t)
    for f in freqs:
        tone = np.sin(2 * np.pi * f * t) + np.sin(2 * np.pi * (f*2) * t) * 0.15
        master_chord += tone
    tremolo = 0.8 + 0.2 * np.sin(2 * np.pi * 5 * t)
    env = np.exp(-1.5 * t)
    return master_chord * tremolo * env * 0.5

# --- MAIN EXECUTION ---
def generate_song():
    if GOOGLE_API_KEY == "YOUR_API_KEY_HERE":
        print("ERROR: Paste your API Key.")
        return

    seed = get_quantum_seed()
    sheet = compose_dual_sections(seed)
    
    if not sheet:
        print("Failed to generate sheet.")
        return

    bpm = sheet.get("bpm", 88)
    swing = sheet.get("swing", 0.1)
    beat_dur = 60 / bpm
    step_len = beat_dur / 4
    
    # --- LOOP CONFIGURATION ---
    total_loops = 10  # 10 loops total
    switch_point = 5  # Switch to "Flavor B" after loop 5
    
    total_samples = int((4 * 4 * beat_dur) * total_loops * SAMPLE_RATE) + 100000
    master = np.zeros(total_samples)
    
    kick_wav = synth_kick_smooth()
    snare_wav = synth_snare_smooth()
    chat_wav = synth_hat_smooth(False)
    ohat_wav = synth_hat_smooth(True)
    
    print(f"Synthesizing {total_loops} loops ({bpm} BPM). Switching flavor at loop {switch_point}...")
    
    for loop in range(total_loops):
        loop_offset_steps = loop * 64
        
        # DECISION: WHICH SECTION TO PLAY?
        if loop < switch_point:
            # First half: Section A (Chill)
            current_data = sheet["section_a"]
        else:
            # Second half: Section B (Spiced up)
            current_data = sheet["section_b"]
            
        # --- PARSE INSTRUMENTS ---
        for step in current_data.get("kick", []):
            t_sec = (step + loop_offset_steps) * step_len
            if (step + loop_offset_steps) % 2 != 0: t_sec += swing
            mix(master, kick_wav, int(t_sec * SAMPLE_RATE))
            
        for step in current_data.get("snare", []):
            t_sec = (step + loop_offset_steps) * step_len
            if (step + loop_offset_steps) % 2 != 0: t_sec += swing
            mix(master, snare_wav, int(t_sec * SAMPLE_RATE))
            
        for step in current_data.get("closed_hat", []):
            t_sec = (step + loop_offset_steps) * step_len
            if (step + loop_offset_steps) % 2 != 0: t_sec += swing
            hum_vol = 0.5 + (np.random.rand() * 0.2)
            mix(master, chat_wav, int(t_sec * SAMPLE_RATE), vol=hum_vol)

        for step in current_data.get("open_hat", []):
            t_sec = (step + loop_offset_steps) * step_len
            if (step + loop_offset_steps) % 2 != 0: t_sec += swing
            mix(master, ohat_wav, int(t_sec * SAMPLE_RATE), vol=0.6)

        for b in current_data.get("bass", []):
            t_sec = (b["step"] + loop_offset_steps) * step_len
            if (b["step"] + loop_offset_steps) % 2 != 0: t_sec += swing
            b_wav = synth_bass_warm(b["freq"], b["dur"], step_len)
            mix(master, b_wav, int(t_sec * SAMPLE_RATE), vol=0.85)

        for k in current_data.get("keys", []):
            t_sec = (k["step"] + loop_offset_steps) * step_len
            if (k["step"] + loop_offset_steps) % 2 != 0: t_sec += swing
            k_wav = synth_keys_lush(k["chord_freqs"], k["dur"], step_len)
            mix(master, k_wav, int(t_sec * SAMPLE_RATE), vol=0.8)

    print("Mastering...")
    master = np.tanh(master * 1.1) 
    m_max = np.max(np.abs(master))
    if m_max > 0: master /= m_max
    
    filename = f"dual_flavor_beat_{seed}.wav"
    wavfile.write(filename, SAMPLE_RATE, (master * 32767).astype(np.int16))
    print(f"DONE! Saved as: {filename}")

if __name__ == "__main__":
    generate_song()
