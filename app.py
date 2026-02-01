import numpy as np
import wave
import json
import collections
import secrets  # Cryptographic strength entropy
import datetime
import random
from openai import OpenAI

# --- 1. STUDIO CONFIGURATION ---
SAMPLE_RATE = 44100
BPM = 118  # "Mid-Tempo" Cyberpunk
BARS_TO_RENDER = 16 

# UPDATED URL: Using 127.0.0.1 is more reliable than localhost
# We keep '/v1' because the OpenAI library expects this structure.
LM_STUDIO_URL = "http://127.0.0.1:1234/v1"

# Standard 16th note length (before swing)
BASE_STEP_SAMPLES = int((60.0 / BPM / 4.0) * SAMPLE_RATE)

# --- 2. HUMAN TIMING ENGINE (GROOVE) ---
class HumanClock:
    def __init__(self):
        self.sys_random = secrets.SystemRandom()
        # 0.15 = 15% Swing (Classic "MPC 3000" feel)
        self.swing_amount = 0.15 

    def get_step_length(self, step_index):
        """
        Calculates sample duration for a specific 16th note step.
        Applies Swing (Long-Short-Long-Short) + Micro-Jitter.
        """
        # SWING LOGIC
        # Even steps (0, 2, 4...) -> Longer ("On" beat)
        # Odd steps (1, 3, 5...) -> Shorter ("Off" beat)
        if step_index % 2 == 0:
            swing_offset = int(BASE_STEP_SAMPLES * self.swing_amount)
        else:
            swing_offset = -int(BASE_STEP_SAMPLES * self.swing_amount)
            
        # MICRO-JITTER (Human Imperfection)
        # +/- 40 samples is ~1ms. Just enough to feel "loose".
        jitter = self.sys_random.randint(-40, 40)
        
        return BASE_STEP_SAMPLES + swing_offset + jitter

# --- 3. DYNAMIC DRUM MODULE (ROUND ROBIN) ---
class HumanDrumModule:
    def __init__(self):
        print(">>> Synthesizing Drum Variations (Round Robin)...")
        # Generate 4 slight variations of the Kick
        self.kicks = [self._make_kick() for _ in range(4)]
        # Generate 8 distinct variations of the Snare
        self.snares = [self._make_snare() for _ in range(8)]

    def _make_kick(self):
        # Kick: Sine sweep
        decay_var = np.random.uniform(14, 16) # Subtle timbre shift
        t = np.linspace(0, 0.3, int(SAMPLE_RATE * 0.3))
        freq = 150 * np.exp(-decay_var * t) + 40
        audio = np.sin(2 * np.pi * np.cumsum(freq) / SAMPLE_RATE)
        # Punchy envelope
        return (audio * np.exp(-6 * t) * 0.9).astype(np.float32)

    def _make_snare(self):
        # Snare: Noise + Tone
        t = np.linspace(0, 0.25, int(SAMPLE_RATE * 0.25))
        # New random noise seed for every snare = organic sound
        noise = np.random.uniform(-1, 1, len(t)) 
        # Slight pitch variation in the "body" of the snare
        tone_freq = np.random.uniform(175, 185)
        tone = np.sin(2 * np.pi * tone_freq * t) * 0.3
        return ((noise * 0.6 + tone) * np.exp(-12 * t) * 0.6).astype(np.float32)

    def get_kick(self):
        return secrets.choice(self.kicks)

    def get_snare(self):
        return secrets.choice(self.snares)

# --- 4. SYNTHESIS MODULE ---
class SynthModule:
    def render_note(self, midi_note, duration_samples, params, type="bass"):
        freq = 440.0 * (2.0**((midi_note - 69.0) / 12.0))
        t = np.arange(duration_samples) / SAMPLE_RATE
        
        # OSCILLATOR
        if type == "bass": 
            # Sawtooth (Aggressive)
            osc = ((t * freq) % 1.0) * 2.0 - 1.0
        else: 
            # Triangle (Hollow/Eerie)
            osc = (2/np.pi) * np.arcsin(np.sin(2 * np.pi * freq * t))
            # Add simple vibrato to lead
            vibrato = np.sin(2 * np.pi * 5.0 * t) * 0.005
            osc = (2/np.pi) * np.arcsin(np.sin(2 * np.pi * freq * (t + vibrato)))
            
        # ENVELOPE (ADSR)
        env = np.zeros(duration_samples)
        attack_len = int(params.get('attack', 0.01) * SAMPLE_RATE)
        decay_len = int(params.get('decay', 0.1) * SAMPLE_RATE)
        sustain_lvl = params.get('sustain', 0.5)
        
        # Safety checks for short notes
        if attack_len >= duration_samples: 
            attack_len = duration_samples
            decay_len = 0
        elif attack_len + decay_len > duration_samples:
            decay_len = duration_samples - attack_len
            
        env[:attack_len] = np.linspace(0, 1, attack_len)
        if decay_len > 0:
            env[attack_len:attack_len+decay_len] = np.linspace(1, sustain_lvl, decay_len)
            env[attack_len+decay_len:] = sustain_lvl
            
        # Release fade
        release_len = min(500, duration_samples)
        env[-release_len:] *= np.linspace(1, 0, release_len)

        return osc * env

# --- 5. THE AI STUDIO RENDERER ---
class StudioRenderer:
    def __init__(self):
        # Using the updated URL
        self.client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")
        self.clock = HumanClock()
        self.drums = HumanDrumModule()
        self.synth = SynthModule()
        self.history = collections.deque(maxlen=4)
        self.song_buffer = []

    def get_ai_bar(self, bar_idx):
        """Query Llama 3 for the next bar's composition"""
        prompt = (
            "You are a Cyberpunk Music Composer. Output JSON ONLY.\n"
            "Create a 16-step pattern. Use a dark, driving progression.\n"
            "Format: {"
            "\"root\":\"C2\", "
            "\"kick\":\"x---x---x---x---\", "
            "\"snare\":\"----x-------x---\", "
            "\"bass_pat\":\"x-x-x-x-x-x-x-x-\", "
            "\"lead_note\":\"G4\", "
            "\"lead_pat\":\"-------x-------x\", "
            "\"cutoff\": 2000"
            "}\n"
            f"Context: Bar {bar_idx + 1} of {BARS_TO_RENDER}. Intensity: High."
        )
        
        try:
            print(f"-> AI Thinking (Bar {bar_idx+1})...")
            resp = self.client.chat.completions.create(
                model="model-identifier",
                messages=[{"role": "system", "content": prompt}],
                temperature=0.9, max_tokens=350, stop=["}"]
            )
            raw = resp.choices[0].message.content
            if "}" not in raw: raw += "}"
            json_str = raw[raw.find('{'):raw.rfind('}')+1]
            data = json.loads(json_str)
            self.history.append(data.get("root", "C2"))
            return data
        except Exception as e:
            print(f"AI Glitch ({e}). Using backup pattern.")
            return {
                "root": "C2", "kick": "x---x---x---x---", "snare": "----x-------x---",
                "bass_pat": "xxxxxxxxxxxxxxxx", "lead_pat": "----------------", "cutoff": 1500
            }

    def render_song(self):
        print(f"/// INITIALIZING RENDER: {BARS_TO_RENDER} BARS @ {BPM} BPM ///")
        print(f"/// TARGETING: {LM_STUDIO_URL} ///")
        
        note_map = {
            'C2':36, 'C#2':37, 'D2':38, 'D#2':39, 'E2':40, 'F2':41, 'F#2':42, 'G2':43, 'A2':45, 
            'C3':48, 'C4':60, 'G4':67, 'C5':72, 'F4':65, 'Bb4':70
        }
        
        # Filter State
        filter_y1 = 0.0

        for bar_idx in range(BARS_TO_RENDER):
            data = self.get_ai_bar(bar_idx)
            
            # Parse Parameters
            kick_pat = data.get('kick', '')
            snare_pat = data.get('snare', '')
            bass_pat = data.get('bass_pat', '')
            lead_pat = data.get('lead_pat', '')
            
            root = note_map.get(data.get('root', 'C2'), 36)
            lead_note = note_map.get(data.get('lead_note', 'G4'), 67)
            cutoff = float(data.get('cutoff', 2000))

            # RENDER 16 STEPS
            for step in range(16):
                # 1. TIMING: Get Swing-adjusted length
                step_len = self.clock.get_step_length(step)
                mix_chunk = np.zeros(step_len)
                
                # 2. DYNAMICS: Determine Velocity
                if step % 4 == 0:
                    velocity = np.random.uniform(0.95, 1.0)
                else:
                    velocity = np.random.uniform(0.75, 0.9)

                # 3. DRUMS
                if step < len(kick_pat) and kick_pat[step] in 'xX':
                    sample = self.drums.get_kick()
                    slen = min(step_len, len(sample))
                    mix_chunk[:slen] += sample[:slen] * velocity

                if step < len(snare_pat) and snare_pat[step] in 'xX':
                    sample = self.drums.get_snare()
                    slen = min(step_len, len(sample))
                    mix_chunk[:slen] += sample[:slen] * velocity * 0.9

                # 4. SYNTHS
                synth_chunk = np.zeros(step_len)
                
                if step < len(bass_pat) and bass_pat[step] in 'xX':
                    note = self.synth.render_note(root, step_len, 
                           {'attack':0.01, 'decay':0.15, 'sustain':0.0}, "bass")
                    synth_chunk += note * 0.6 * velocity

                if step < len(lead_pat) and lead_pat[step] in 'xX':
                    note = self.synth.render_note(lead_note, step_len, 
                           {'attack':0.1, 'decay':0.4, 'sustain':0.4}, "lead")
                    synth_chunk += note * 0.4 * velocity

                # 5. FILTER
                alpha = (2 * np.pi * cutoff / SAMPLE_RATE)
                for i in range(step_len):
                    filter_y1 = filter_y1 + alpha * (synth_chunk[i] - filter_y1)
                    synth_chunk[i] = filter_y1
                
                mix_chunk += synth_chunk
                self.song_buffer.append(mix_chunk)

        self.save_wav()

    def save_wav(self):
        filename = f"HUMAN_AI_JAM_{datetime.datetime.now().strftime('%H%M%S')}.wav"
        print(f"/// EXPORTING TO {filename} ...")
        
        full_audio = np.concatenate(self.song_buffer)
        
        # LIMITER
        max_val = np.max(np.abs(full_audio))
        if max_val > 0:
            full_audio = full_audio / max_val * 0.95
            
        int_audio = (full_audio * 32767).astype(np.int16)
        
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1) 
            wf.setsampwidth(2) 
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(int_audio.tobytes())
            
        print(f"/// DONE. PLAY {filename} ///")

if __name__ == "__main__":
    studio = StudioRenderer()
    studio.render_song()
