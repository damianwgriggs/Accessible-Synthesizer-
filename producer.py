import numpy as np
import wave
import struct
import random
import tkinter as tk
from tkinter import filedialog, simpledialog

# --- CONFIGURATION ---
SAMPLE_RATE = 44100
BPM = 92  # Classic 90s Hip Hop Tempo
SWING = 0.03 # The "Funk" Factor (Human timing offset)

# --- 1. DRUM SYNTHESIS (THE KIT) ---

def make_kick(dur=0.15):
    # Deep, punchy 808 style kick
    t = np.arange(int(dur * SAMPLE_RATE)) / SAMPLE_RATE
    freq = np.linspace(120, 40, len(t)) # Pitch drop
    amp = np.linspace(1.0, 0.0, len(t))**2
    audio = np.sin(2 * np.pi * freq * t) * amp
    # Add a little distortion (clipping) for grit
    return np.clip(audio * 1.5, -1.0, 1.0)

def make_snare(dur=0.12):
    # Sharp, tight snare
    n = int(dur * SAMPLE_RATE)
    noise = np.random.uniform(-1, 1, n)
    tone_t = np.arange(n) / SAMPLE_RATE
    tone = np.sin(2 * np.pi * 180 * tone_t) * (np.linspace(1, 0, n)**4)
    env = np.linspace(1.0, 0.0, n)**2
    # Mix noise (rattle) and tone (body)
    return (noise * 0.7 + tone * 0.3) * env

def make_hat(dur=0.04, open_hat=False):
    # Metallic high frequency
    n = int(dur * SAMPLE_RATE)
    noise = np.random.uniform(-1, 1, n)
    # High pass filter simulation (alternating signs flip frequency)
    for i in range(1, n, 2): noise[i] *= -1 
    
    decay = 1.0 if open_hat else 8.0 # Open hats ring out longer
    env = np.linspace(1.0, 0.0, n)**decay
    return noise * env * 0.4

# --- 2. THE FUNK PATTERN GENERATOR ---

def get_pattern(section_type):
    # Returns a 1-bar grid (16 steps). 
    # 1 = Kick, 2 = Snare, 3 = Closed Hat, 4 = Open Hat, 5 = Ghost Snare
    
    # Empty grid of 16 steps (16th notes)
    grid = [[] for _ in range(16)]
    
    # --- BASE RHYTHM (ALL SECTIONS) ---
    # Hats on every 8th note
    for i in range(0, 16, 2): 
        grid[i].append(3) 
    
    if section_type == 'INTRO':
        # Just Hats
        return grid
        
    if section_type == 'VERSE':
        # Sparse, Groovy
        grid[0].append(1)  # Kick
        grid[4].append(2)  # Snare
        grid[10].append(1) # Kick (Syncopated)
        grid[12].append(2) # Snare
        # Ghost notes for funk
        grid[7].append(5)
        grid[9].append(5)

    elif section_type == 'CHORUS':
        # Heavy, Full
        grid[0].append(1)
        grid[2].append(1)
        grid[4].append(2)
        grid[8].append(1)
        grid[12].append(2)
        grid[14].append(4) # Open Hat at end
        # Busy Hats
        for i in range(1, 16, 2): grid[i].append(3)

    elif section_type == 'BRIDGE':
        # Breakdown - No Kick
        grid[4].append(2)
        grid[12].append(2)
        grid[0].append(4) # Open Hat start
        
    return grid

# --- 3. THE ARRANGER ---

def build_song(user_loop_file):
    print("Synthesizing Drums...")
    kick = make_kick()
    snare = make_snare()
    hat_c = make_hat(0.04, False)
    hat_o = make_hat(0.2, True)
    ghost = make_snare() * 0.3 # Quiet snare
    
    # Load User Audio
    wf = wave.open(user_loop_file, 'rb')
    user_data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32767.0
    wf.close()
    
    # --- SONG STRUCTURE DEFINITION ---
    # 4 bars Intro, 16 bars Verse, 8 bars Chorus, 16 bars Verse, 8 bars Chorus, 8 bars Bridge, 8 bars Outro
    structure = ['INTRO']*4 + ['VERSE']*16 + ['CHORUS']*8 + ['VERSE']*16 + ['CHORUS']*8 + ['BRIDGE']*8 + ['INTRO']*8
    
    samples_per_beat = int(60 / BPM * SAMPLE_RATE)
    samples_per_16th = int(samples_per_beat / 4)
    bar_length = samples_per_beat * 4
    
    total_samples = len(structure) * bar_length
    master_mix = np.zeros(total_samples, dtype=np.float32)
    
    print("Arranging Track...")
    
    # 1. LAY DOWN THE DRUMS
    cursor = 0
    for section in structure:
        pattern = get_pattern(section)
        
        for step in range(16):
            # CALCULATE SWING
            # If it's an even numbered step (off-beat), push it late
            swing_offset = int(samples_per_beat * SWING) if step % 2 != 0 else 0
            
            step_start = cursor + (step * samples_per_16th) + swing_offset
            
            sounds_to_play = pattern[step]
            for sound_id in sounds_to_play:
                audio = None
                if sound_id == 1: audio = kick
                elif sound_id == 2: audio = snare
                elif sound_id == 3: audio = hat_c
                elif sound_id == 4: audio = hat_o
                elif sound_id == 5: audio = ghost
                
                if audio is not None:
                    # Mix safely
                    end = step_start + len(audio)
                    if end < total_samples:
                        master_mix[step_start:end] += audio
        
        cursor += bar_length

    # 2. LAYER THE USER LOOP
    # We tile the user loop over the whole song
    print("Layering Synth Loop...")
    user_cursor = 0
    while user_cursor < total_samples:
        chunk_len = len(user_data)
        if user_cursor + chunk_len > total_samples:
            chunk_len = total_samples - user_cursor
            
        # Volume Automation based on section?
        # Let's make the synth slightly quieter during Verses to leave room for Rapping
        # This is hard to do perfectly without a grid map, so we'll keep it steady for now
        # but apply a Master Limiter later.
        
        master_mix[user_cursor : user_cursor+chunk_len] += user_data[:chunk_len] * 0.75
        user_cursor += len(user_data)

    # 3. MASTERING (Limiter)
    print("Mastering...")
    # Soft clip to glue drums and synth together
    master_mix = np.tanh(master_mix * 1.2) 
    
    # 4. EXPORT
    out_name = user_loop_file.replace(".wav", "_FUNK_FUSION_SONG.wav")
    audio_int16 = (master_mix * 32767).astype(np.int16)
    
    with wave.open(out_name, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())
        
    print(f"SUCCESS! Created: {out_name}")

# --- GUI ---
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    print("Select your Synth Loop...")
    f = filedialog.askopenfilename(filetypes=[("WAV", "*.wav")])
    if f: build_song(f)
