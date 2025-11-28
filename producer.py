import numpy as np
import wave
import struct
import random
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os

# --- CONFIG ---
SAMPLE_RATE = 44100

# --- 1. THE BRAIN (DSP ANALYSIS) ---

def analyze_audio(file_path):
    print(f"Analyzing {file_path}...")
    
    wf = wave.open(file_path, 'rb')
    n_frames = wf.getnframes()
    raw_data = wf.readframes(n_frames)
    wf.close()
    
    # Convert to float array
    audio = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32767.0
    duration = n_frames / SAMPLE_RATE
    
    # --- A. BPM DETECTION (Math based on Loop Length) ---
    # We assume the user recorded a perfect 4-bar or 2-bar loop.
    # Formula: BPM = (Beats / Seconds) * 60
    
    # Hypothesis 1: It's 2 bars (8 beats)
    bpm_2bar = (8 / duration) * 60
    # Hypothesis 2: It's 4 bars (16 beats)
    bpm_4bar = (16 / duration) * 60
    
    # Logic: Most Hip Hop is between 70 and 110 BPM.
    # We pick the calculated BPM that falls closest to this range.
    if 80 <= bpm_4bar <= 160:
        detected_bpm = bpm_4bar
        bars = 4
    else:
        detected_bpm = bpm_2bar
        bars = 2
        
    # Clamp extreme values
    if detected_bpm < 60: detected_bpm *= 2
    if detected_bpm > 180: detected_bpm /= 2
    
    # --- B. PITCH DETECTION (Fast Fourier Transform) ---
    # We analyze the first 1 second to find the Root Note
    chunk_size = min(len(audio), SAMPLE_RATE) # 1 second
    chunk = audio[:chunk_size]
    
    # Hanning window to smooth edges
    windowed = chunk * np.hanning(len(chunk))
    # Perform FFT
    spectrum = np.fft.rfft(windowed)
    frequencies = np.fft.rfftfreq(len(chunk), 1/SAMPLE_RATE)
    
    # Find the strongest frequency (The Fundamental)
    # We ignore very low rumble (< 40Hz)
    magnitude = np.abs(spectrum)
    magnitude[frequencies < 40] = 0 
    peak_index = np.argmax(magnitude)
    detected_freq = frequencies[peak_index]
    
    print(f"Analysis Complete: {detected_bpm:.2f} BPM | Pitch: {detected_freq:.2f} Hz")
    return detected_bpm, detected_freq, audio

# --- 2. ADAPTIVE DRUM SYNTHESIS ---

def smart_kick(freq, duration_sec, sr=SAMPLE_RATE):
    # Tunes the kick to the detected frequency
    
    # If freq is high (e.g. 440Hz), drop it down octaves until it's bass (40-60Hz)
    target_freq = freq
    while target_freq > 65:
        target_freq /= 2.0
    while target_freq < 40:
        target_freq *= 2.0
        
    t = np.arange(int(duration_sec * sr)) / sr
    
    # Pitch envelope: Start high (punch) drop to target (hum)
    freq_sweep = np.linspace(target_freq * 4, target_freq, len(t))
    
    # Amplitude: Punchy attack, long tail
    amp = np.linspace(1.0, 0.0, len(t))**2
    
    audio = np.sin(2 * np.pi * freq_sweep * t) * amp
    
    # Distortion for "Modern" sound
    return np.clip(audio * 1.8, -1.0, 1.0)

def smart_hat(bpm, duration, open_hat=False):
    # If BPM is slow, we make hats shorter/tighter. 
    # If BPM is fast, we make them softer.
    
    decay_sharpness = 4 if bpm < 100 else 8 # Tighter hats for fast beats
    if open_hat: decay_sharpness = 1
    
    n = int(duration * SAMPLE_RATE)
    noise = np.random.uniform(-1, 1, n)
    env = np.linspace(1.0, 0.0, n)**decay_sharpness
    
    # High Pass Filter hack
    for i in range(1, n, 2): noise[i] *= -1
        
    return noise * env * 0.35

def smart_snare(duration=0.15):
    # Standard crisp snare
    n = int(duration * SAMPLE_RATE)
    noise = np.random.uniform(-1, 1, n) * (np.linspace(1,0,n)**2)
    tone = np.sin(np.linspace(200, 100, n) * 2 * np.pi * (np.arange(n)/SAMPLE_RATE)) * (np.linspace(1,0,n)**4)
    return (noise*0.6 + tone*0.4)

# --- 3. INTELLIGENT SEQUENCER ---

def generate_pattern(section, bpm):
    # Returns 16-step grid based on BPM "Vibe"
    grid = [[] for _ in range(16)]
    
    # HATS LOGIC
    if bpm < 85:
        # TRAP STYLE: Double time hats (32nd notes simulated)
        for i in range(16): grid[i].append(3) 
    else:
        # BOOM BAP STYLE: 8th note hats
        for i in range(0, 16, 2): grid[i].append(3)

    if section == 'INTRO': return grid
    
    # KICK/SNARE LOGIC
    # Snare always on 4 and 12 (Beats 2 and 4)
    grid[4].append(2)
    grid[12].append(2)
    
    if section == 'CHORUS':
        # Busy Kick
        grid[0].append(1)
        grid[2].append(1) # Double kick start
        grid[10].append(1)
        if bpm < 95: grid[14].append(1) # Extra kick for slow jams
        grid[14].append(4) # Open hat end
    elif section == 'VERSE':
        # Spacious Kick
        grid[0].append(1)
        grid[10].append(1)
        
    return grid

# --- 4. MAIN PROCESSOR ---

def process_smart_song(file_path):
    # A. ANALYZE
    bpm, pitch, user_loop = analyze_audio(file_path)
    
    # B. SYNTHESIZE DRUMS (TUNED)
    kick = smart_kick(pitch, 0.3) # <--- TUNED TO YOUR LOOP
    snare = smart_snare()
    hat_c = smart_hat(bpm, 0.05, False)
    hat_o = smart_hat(bpm, 0.3, True)
    
    # C. ARRANGE
    samples_per_beat = int(60 / bpm * SAMPLE_RATE)
    samples_per_16th = int(samples_per_beat / 4)
    bar_length = samples_per_beat * 4
    
    # Structure: Intro(4) -> Verse(16) -> Chorus(8) -> Verse(16) -> Outro(4)
    structure = ['INTRO']*4 + ['VERSE']*16 + ['CHORUS']*8 + ['VERSE']*16 + ['INTRO']*4
    
    total_len = len(structure) * bar_length
    mix = np.zeros(total_len, dtype=np.float32)
    
    # D. BUILD BEAT
    cursor = 0
    print(f"Constructing arrangement at {bpm:.1f} BPM...")
    
    for section in structure:
        pattern = generate_pattern(section, bpm)
        
        for step in range(16):
            # Swing Math: Delay every even step by a calculated percentage
            swing_samps = int(samples_per_16th * 0.3) if step%2!=0 else 0
            
            start = cursor + (step * samples_per_16th) + swing_samps
            
            for sound_id in pattern[step]:
                s = None
                if sound_id==1: s=kick
                elif sound_id==2: s=snare
                elif sound_id==3: s=hat_c
                elif sound_id==4: s=hat_o
                
                if s is not None:
                    end = start + len(s)
                    if end < total_len: mix[start:end] += s
        
        cursor += bar_length
        
    # E. LAYER USER LOOP (TIME STRETCHING SIMULATION)
    # We assume the user loop is "roughly" correct and tile it.
    # A true time-stretch requires complex libraries, so we use "Re-triggering"
    # We re-trigger the loop at the start of every bar (or every X bars) to keep it in sync.
    
    loop_dur_samples = len(user_loop)
    # How many bars is the loop roughly?
    bars_in_loop = round(loop_dur_samples / bar_length)
    if bars_in_loop == 0: bars_in_loop = 1
    
    loop_trigger_interval = bar_length * bars_in_loop
    
    curr = 0
    while curr < total_len:
        # Safety check for mix length
        end = curr + len(user_loop)
        if end > total_len:
            end = total_len
            chunk = user_loop[:end-curr]
        else:
            chunk = user_loop
            
        mix[curr:curr+len(chunk)] += chunk * 0.8
        curr += loop_trigger_interval # Sync point
        
    # F. FINALIZE
    out_name = file_path.replace(".wav", f"_SMART_MIX_{int(bpm)}BPM.wav")
    
    # Limiter
    mix = np.tanh(mix * 1.5)
    
    # Save
    d = (mix * 32767).astype(np.int16)
    with wave.open(out_name, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SAMPLE_RATE)
        w.writeframes(d.tobytes())
        
    return out_name, bpm, pitch

# --- 5. UI ---
class SmartProducerUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("THE SMART PRODUCER")
        self.geometry("600x400")
        self.configure(bg='black')
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('.', background='black', foreground='white', font=('Arial', 14))
        style.configure('TButton', background='#333333', foreground='white', borderwidth=0)
        
        ttk.Label(self, text="AI MATH PRODUCER", font=('Arial', 24, 'bold')).pack(pady=30)
        
        self.status = tk.StringVar(value="Ready to analyze.")
        ttk.Label(self, textvariable=self.status, foreground='#00ff00').pack(pady=20)
        
        ttk.Button(self, text="LOAD LOOP & GENERATE", command=self.run).pack(ipadx=20, ipady=20)
        
    def run(self):
        f = filedialog.askopenfilename(filetypes=[("WAV", "*.wav")])
        if f:
            self.status.set("Analyzing Physics...")
            self.update()
            try:
                out, bpm, pitch = process_smart_song(f)
                self.status.set(f"Done! {bpm:.1f} BPM | Tuned to {pitch:.1f}Hz\nSaved as: {os.path.basename(out)}")
            except Exception as e:
                self.status.set(f"Error: {e}")

if __name__ == "__main__":
    app = SmartProducerUI()
    app.mainloop()
