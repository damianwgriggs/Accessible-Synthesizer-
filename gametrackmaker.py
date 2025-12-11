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
GOOGLE_API_KEY = "YOURAPIKEYHERE"

MODEL_NAME = 'gemini-2.5-flash'
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

SAMPLE_RATE = 44100
BIT_CRUSH_ENABLED = True 

# --- 1. CHAOS ENGINE ---
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

# --- 2. THE COMPOSER ---
def compose_continuous_track(seed):
    print(f"Contacting {MODEL_NAME} for Seamless Track...")
    
    prompt = f"""
    You are a Video Game Composer. Seed: {seed}.
    Create a BUSY, CONTINUOUS Retro Game Soundtrack (3 Phases).
    
    JSON Structure:
    {{
      "bpm": integer (115-135),
      "global_key_freq": float (root note frequency, e.g. 55.0),
      "phase_1": {{ "kick": [], "snare": [], "closed_hat": [], "open_hat": [], "bass": [], "keys": [] }},
      "phase_2": {{ "kick": [], "snare": [], "closed_hat": [], "open_hat": [], "bass": [], "keys": [] }},
      "phase_3": {{ "kick": [], "snare": [], "closed_hat": [], "open_hat": [], "bass": [], "keys": [] }}
    }}
    
    Directives:
    1. **Fill the Loop**: Ensure notes exist in steps 0-32 AND steps 33-63.
    2. **Phase 1**: Driving Bass. 
    3. **Phase 3**: Maximum Intensity.
    4. **Drums**: List of step numbers only [0, 4, 8...].
    5. **Bass/Keys**: [ {{ "step": int, "freq": float, "dur": int }} ]
    """
    try:
        response = model.generate_content(prompt)
        text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        print(f"Gemini Error: {e}")
        return None

# --- 3. DENSITY ENFORCER (The Fix) ---
def ensure_full_loop_density(data, phase_name):
    """
    Checks if the loop dies out halfway through (steps 32-63).
    If empty, mirrors the first half to the second half.
    """
    
    # 1. Check Drums
    for drum in ["kick", "snare", "closed_hat"]:
        steps = data.get(drum, [])
        # If we have no hits in the second half (steps >= 32)
        if not any(s >= 32 for s in steps):
            # Mirror the first half beats to the second half
            new_hits = [s + 32 for s in steps if s < 32]
            data[drum] = steps + new_hits
            
    # 2. Check Bass/Keys (Dictionary Lists)
    for inst in ["bass", "keys"]:
        notes = data.get(inst, [])
        if not any(n.get("step", 0) >= 32 for n in notes):
            new_notes = []
            for n in notes:
                if n.get("step", 0) < 32:
                    # Create a copy shifted by 32 steps
                    copy_note = n.copy()
                    copy_note["step"] = n.get("step", 0) + 32
                    new_notes.append(copy_note)
            data[inst] = notes + new_notes

    # 3. Last Resort Fallbacks (If completely empty)
    if len(data.get("closed_hat", [])) < 8:
        data["closed_hat"] = list(range(0, 64, 4)) # Quarter notes
    
    if len(data.get("kick", [])) < 2:
        data["kick"] = [0, 32] # Downbeats
        
    return data

# --- 4. DSP INSTRUMENTS ---

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

def apply_bitcrusher(audio):
    rate_reduction = 2 
    for i in range(0, len(audio), rate_reduction):
        audio[i:i+rate_reduction] = audio[i]
    return audio

def synth_pad_drone(freq, duration):
    # LOUDER GLUE: Increased volume to 0.25
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration))
    wave = np.sin(2 * np.pi * freq * t) + (np.sign(np.sin(2 * np.pi * (freq * 1.01) * t)) * 0.3)
    lfo = 0.8 + 0.2 * np.sin(2 * np.pi * 0.5 * t)
    return wave * lfo * 0.25 

def synth_kick_retro():
    dur = 0.4
    t = np.linspace(0, dur, int(SAMPLE_RATE * dur))
    freq = 150 * np.exp(-12 * t) + 40
    sub = np.sin(2 * np.pi * freq * t)
    env = np.exp(-4 * t)
    return np.clip(sub * env * 1.5, -0.8, 0.8)

def synth_snare_retro():
    dur = 0.3 
    t = np.linspace(0, dur, int(SAMPLE_RATE * dur))
    noise = np.random.uniform(-1, 1, len(t)) * np.exp(-8 * t)
    tone = np.sign(np.sin(2 * np.pi * 200 * t)) * np.exp(-6 * t) * 0.4 
    return (noise + tone) * 0.7

def synth_hat_retro(open_h=False):
    dur = 0.4 if open_h else 0.1
    t = np.linspace(0, dur, int(SAMPLE_RATE * dur))
    noise = np.random.uniform(-1, 1, len(t))
    env = np.exp(-8 * t) if open_h else np.exp(-30 * t)
    return noise * env * 0.4

def synth_bass_retro(freq, steps, step_len):
    steps = max(steps, 4) 
    dur = steps * step_len
    t = np.linspace(0, dur, int(SAMPLE_RATE * dur))
    wave = np.sign(np.sin(2 * np.pi * freq * t))
    env = np.ones_like(t)
    env[-2000:] = np.linspace(1, 0, 2000)
    return wave * env * 0.6

def synth_keys_retro(freqs, steps, step_len):
    steps = max(steps, 4)
    dur = steps * step_len
    t = np.linspace(0, dur, int(SAMPLE_RATE * dur))
    master = np.zeros_like(t)
    for f in freqs:
        tone = np.where(np.sin(2 * np.pi * f * t) > 0.5, 1.0, -1.0)
        master += tone
    lfo = 0.7 + 0.3 * np.sin(2 * np.pi * 8 * t)
    return master * lfo * 0.35

# --- MAIN GENERATOR ---
def generate_game_soundtrack():
    if GOOGLE_API_KEY == "YOUR_API_KEY_HERE":
        print("ERROR: Paste your API Key.")
        return

    seed = get_quantum_seed()
    sheet = compose_continuous_track(seed)
    if not sheet: return

    bpm = sheet.get("bpm", 120)
    swing = 0.0 # Swing can cause gaps in retro loops, keeping it straight (0.0) is safer
    root_freq = sheet.get("global_key_freq", 55.0)
    
    beat_dur = 60 / bpm
    step_len = beat_dur / 4
    
    # Structure
    structure = [
        ("phase_1", 4),
        ("phase_2", 4),
        ("phase_3", 8),
        ("phase_2", 4),
        ("phase_3", 8),
        ("phase_1", 4)
    ]
    
    total_loops = sum(loops for _, loops in structure)
    print(f"Arranging Seamless Track: {total_loops} bars ({bpm} BPM).")

    # 1. Calculate Exact Duration (No Padding)
    exact_duration_samples = int(total_loops * 4 * beat_dur * SAMPLE_RATE)
    
    # 2. Create Buffer with slight padding for mix tail overflow
    master = np.zeros(exact_duration_samples + 44100) 
    
    print("Layering background 'Glue'...")
    total_dur_sec = total_loops * 4 * beat_dur
    mix(master, synth_pad_drone(root_freq, total_dur_sec), 0)
    mix(master, synth_pad_drone(root_freq * 1.5, total_dur_sec), 0)
    
    kick = synth_kick_retro()
    snare = synth_snare_retro()
    chat = synth_hat_retro(False)
    ohat = synth_hat_retro(True)
    
    current_loop_index = 0
    
    for phase_name, repeat_count in structure:
        print(f" -> Rendering {phase_name} ({repeat_count} bars)...")
        
        raw_data = sheet.get(phase_name, {})
        # FIX: Ensure density spans the FULL loop (0-63)
        data = ensure_full_loop_density(raw_data, phase_name)
        
        for _ in range(repeat_count):
            loop_offset_steps = current_loop_index * 64
            
            # Helper to safely get step int
            def get_step(val):
                if isinstance(val, dict): return val.get("step", 0)
                return val

            for s in data.get("kick", []):
                t = (get_step(s) + loop_offset_steps) * step_len
                mix(master, kick, int(t * SAMPLE_RATE))
                
            for s in data.get("snare", []):
                t = (get_step(s) + loop_offset_steps) * step_len
                mix(master, snare, int(t * SAMPLE_RATE))
                
            for s in data.get("closed_hat", []):
                t = (get_step(s) + loop_offset_steps) * step_len
                mix(master, chat, int(t * SAMPLE_RATE), vol=0.6)
                
            for s in data.get("open_hat", []):
                t = (get_step(s) + loop_offset_steps) * step_len
                mix(master, ohat, int(t * SAMPLE_RATE), vol=0.6)
            
            for b in data.get("bass", []):
                t = (b["step"] + loop_offset_steps) * step_len
                mix(master, synth_bass_retro(b.get("freq", root_freq), b.get("dur", 8), step_len), int(t * SAMPLE_RATE))
                
            for k in data.get("keys", []):
                t = (k["step"] + loop_offset_steps) * step_len
                freqs = k.get("chord_freqs") or k.get("freqs") or k.get("frequencies")
                if not freqs and "freq" in k: freqs = [k["freq"]]
                if freqs:
                    mix(master, synth_keys_retro(freqs, k.get("dur", 8), step_len), int(t * SAMPLE_RATE))
            
            current_loop_index += 1

    print("Mastering (Strict Cut)...")
    
    # 3. STRICT CUT: Chop the file EXACTLY where the music math ends
    # This prevents any trailing silence from a buffer mismatch
    master = master[:exact_duration_samples]
        
    if BIT_CRUSH_ENABLED:
        master = apply_bitcrusher(master)
        
    m_max = np.max(np.abs(master))
    if m_max > 0: master /= m_max
    
    filename = f"Seamless_Retro_Track_{seed}.wav"
    wavfile.write(filename, SAMPLE_RATE, (master * 32767).astype(np.int16))
    print(f"DONE! Saved: {filename}")

if __name__ == "__main__":
    generate_game_soundtrack()
