import numpy as np
import wave
import json
import collections
import secrets
import datetime
import random
from openai import OpenAI

# --- 1. GLOBAL CONFIG ---
SAMPLE_RATE = 44100
BPM = 94
# 1:30 duration @ 94 BPM = ~36 bars. We'll render 40 to be safe.
TOTAL_BARS = 40 
LM_STUDIO_URL = "http://127.0.0.1:1234/v1"

BASE_STEP_SAMPLES = int((60.0 / BPM / 4.0) * SAMPLE_RATE)

# --- 2. THE CHOP SHOP (TIMING) ---
class MPCClock:
    def __init__(self):
        self.sys_random = secrets.SystemRandom()
        # Heavy Swing (22%) for that "Dilla" stumbling feel
        self.swing_amount = 0.22 

    def get_step_length(self, step_index):
        # Swing Logic
        if step_index % 2 == 0:
            swing_offset = int(BASE_STEP_SAMPLES * self.swing_amount)
        else:
            swing_offset = -int(BASE_STEP_SAMPLES * self.swing_amount)
        
        # INCREASED RANDOM: "Drunk" Drummer Jitter (+/- 3ms)
        jitter = self.sys_random.randint(-130, 130) 
        return BASE_STEP_SAMPLES + swing_offset + jitter

# --- 3. DRUMS: DIRTY KITS ---
class DirtyDrumModule:
    def __init__(self):
        print(">>> Sampling from Vinyl...")
        self.kicks = [self._make_kick() for _ in range(6)]
        self.snares = [self._make_snare() for _ in range(6)]
        self.hats = [self._make_hat() for _ in range(4)]

    def _make_kick(self):
        # 808 with extra "Dirt" (Noise layer)
        t = np.linspace(0, 0.5, int(SAMPLE_RATE * 0.5))
        freq = 110 * np.exp(-20 * t) + 30
        sine = np.sin(2 * np.pi * np.cumsum(freq) / SAMPLE_RATE)
        # Add grit
        noise = np.random.uniform(-0.1, 0.1, len(t)) * np.exp(-10 * t)
        audio = (sine + noise) * np.exp(-4 * t)
        return np.tanh(audio * 2.0).astype(np.float32) # Distortion

    def _make_snare(self):
        # Layered Clap + Snare
        t = np.linspace(0, 0.2, int(SAMPLE_RATE * 0.2))
        noise = np.random.uniform(-1, 1, len(t)) * np.exp(-12 * t)
        tone = np.sin(2 * np.pi * 200 * t) * np.exp(-15 * t) * 0.4
        # "Clap" transient (multiple bursts)
        clap = np.zeros_like(t)
        for i in [0, 0.01, 0.02]: # 10ms and 20ms delays
            idx = int(i * SAMPLE_RATE)
            if idx < len(clap): clap[idx:] += noise[:-idx] if idx > 0 else noise
        return ((clap * 0.6 + tone) * 0.8).astype(np.float32)

    def _make_hat(self):
        t = np.linspace(0, 0.05, int(SAMPLE_RATE * 0.05))
        return (np.random.uniform(-0.6, 0.6, len(t)) * np.exp(-40 * t)).astype(np.float32)

    def get_kick(self): return secrets.choice(self.kicks)
    def get_snare(self): return secrets.choice(self.snares)
    def get_hat(self): return secrets.choice(self.hats)

# --- 4. SYNTHS: THE CHAOS ENGINE ---
class ChaosSynth:
    def render(self, midi, duration, type="bass"):
        freq = 440.0 * (2.0**((midi - 69.0) / 12.0))
        t = np.arange(duration) / SAMPLE_RATE
        
        # CHAOS FACTOR: Randomize parameters PER NOTE
        detune = random.uniform(0.99, 1.01)
        grit = random.uniform(0.0, 0.2)
        
        if type == "bass":
            # Wobbly Reese Bass
            osc1 = ((t * freq * detune) % 1.0) * 2.0 - 1.0
            osc2 = ((t * freq * (2.0 - detune)) % 1.0) * 2.0 - 1.0
            raw = (osc1 + osc2) * 0.5
            # Bitcrush-ish noise
            raw += np.random.uniform(-grit, grit, len(raw))
            # Lowpass
            audio = np.convolve(raw, np.ones(8)/8, mode='same')
            audio = np.tanh(audio * 3.0) # Hard Clip
            
        else: # Lead
            # Sine with randomized vibrato
            vib_speed = random.uniform(4.0, 7.0)
            vib_depth = random.uniform(0.001, 0.005)
            mod = np.sin(2 * np.pi * vib_speed * t) * vib_depth
            audio = np.sin(2 * np.pi * freq * (t + mod))
            # Decay envelope with random length
            dec_speed = random.uniform(3.0, 6.0)
            audio *= np.exp(-dec_speed * t)

        return audio * 0.6

# --- 5. THE ARRANGER (BRAIN) ---
class SongArranger:
    def __init__(self):
        self.client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")
        self.clock = MPCClock()
        self.drums = DirtyDrumModule()
        self.synth = ChaosSynth()
        
        # MEMORY BANKS
        self.motif_bank = {} # Stores the "Hook" pattern
        self.history = collections.deque(maxlen=8) # Remember last 8 bars

    def get_song_structure(self, bar_idx):
        # STRUCTURE MAP (36-40 Bars total)
        if bar_idx < 4: return "INTRO" # 0-3
        if bar_idx < 20: return "VERSE_1" # 4-19 (16 bars)
        if bar_idx < 28: return "HOOK" # 20-27 (8 bars)
        if bar_idx < 36: return "VERSE_2" # 28-35 (8 bars)
        return "OUTRO"

    def get_pattern(self, bar_idx):
        section = self.get_song_structure(bar_idx)
        
        # MEMORY RECALL: If we are in the HOOK, use the saved pattern!
        if section == "HOOK" and "main_hook" in self.motif_bank:
            print(f"-> Recalling HOOK from Memory...")
            # We add slight variation so it's not a copy-paste robot
            base = self.motif_bank["main_hook"]
            # Occasional variation
            if random.random() > 0.7: 
                base['snare'] = base['snare'].replace('-', 'x', 1) 
            return base

        # AI GENERATION
        prompt = (
            f"Role: Hip Hop Producer. Section: {section}. BPM: {BPM}.\n"
            "Output JSON ONLY. Create a 1-bar loop.\n"
            "Context: Heavy, dark, aggressive.\n"
            "Keys: C Minor, F Minor.\n"
            "Format: {\"root\":\"C2\", \"kick\":\"x--x...\", \"snare\":\"...\", \"bass\":\"x-x...\", \"lead_note\":\"G4\", \"lead_pat\":\"...\"}"
        )
        
        # Specific instructions per section
        if section == "INTRO": prompt += " Sparse drums, long bass notes."
        if section == "VERSE_1": prompt += " Steady boom-bap, room for vocals."
        if section == "HOOK": prompt += " Busy, energetic, catchy lead melody."
        if section == "OUTRO": prompt += " Fade out, decompose the beat."

        try:
            print(f"-> Composing Bar {bar_idx+1} ({section})...")
            resp = self.client.chat.completions.create(
                model="model-identifier",
                messages=[{"role": "system", "content": prompt}],
                temperature=0.95, # High randomness
                max_tokens=350, stop=["}"]
            )
            raw = resp.choices[0].message.content
            if "}" not in raw: raw += "}"
            data = json.loads(raw[raw.find('{'):raw.rfind('}')+1])
            
            # MEMORY SAVE: If this is the first bar of the hook, save it!
            if section == "HOOK" and "main_hook" not in self.motif_bank:
                self.motif_bank["main_hook"] = data
                
            return data
        except:
            return {"root":"C2", "kick":"x---x---x---x---", "snare":"----x-------x---", "bass":"x-x-x-x-x-x-x-x-", "lead_pat":"----------------"}

    def render_full_song(self):
        print(f"/// PRODUCING 1:30 TRACK ({TOTAL_BARS} BARS) ///")
        full_len = int(TOTAL_BARS * 16 * BASE_STEP_SAMPLES * 1.05)
        master_bus = np.zeros(full_len)
        cursor = 0
        
        note_map = {'C2':36, 'C#2':37, 'D2':38, 'Eb2':39, 'E2':40, 'F2':41, 'G2':43, 'Ab2':44, 'A2':45, 'Bb2':46, 'C5':72, 'G4':67, 'F4':65}

        for bar in range(TOTAL_BARS):
            data = self.get_pattern(bar)
            root = note_map.get(data.get('root', 'C2'), 36)
            lead_n = note_map.get(data.get('lead_note', 'G4'), 67)
            
            # The Loop
            for step in range(16):
                step_len = self.clock.get_step_length(step)
                
                # --- RANDOM FILL GENERATOR ---
                # 5% chance of a random drum fill at end of bars
                is_fill = (step > 12 and random.random() < 0.05)
                
                # DRUMS
                vel = random.uniform(0.85, 1.0)
                
                # Kick
                if (step < len(data.get('kick','')) and data['kick'][step] in 'xX') or is_fill:
                    s = self.drums.get_kick()
                    sl = min(len(s), len(master_bus) - cursor)
                    master_bus[cursor:cursor+sl] += s[:sl] * vel

                # Snare
                if (step < len(data.get('snare','')) and data['snare'][step] in 'xX') or is_fill:
                    s = self.drums.get_snare()
                    sl = min(len(s), len(master_bus) - cursor)
                    master_bus[cursor:cursor+sl] += s[:sl] * vel

                # Hats (Ghost notes + Accents)
                if random.random() > 0.1: # Skip 10% for "human error"
                    h = self.drums.get_hat()
                    hl = min(len(h), len(master_bus) - cursor)
                    h_vel = 0.5 if step % 2 == 0 else 0.2
                    # Random open hat accent
                    if random.random() < 0.05: h_vel = 0.8 
                    master_bus[cursor:cursor+hl] += h[:hl] * h_vel

                # SYNTHS
                # Bass
                if step < len(data.get('bass','')) and data['bass'][step] in 'xX':
                    b = self.synth.render(root, step_len * 2, "bass")
                    sl = min(len(b), len(master_bus) - cursor)
                    master_bus[cursor:cursor+sl] += b[:sl] * 0.9

                # Lead
                if step < len(data.get('lead_pat','')) and data['lead_pat'][step] in 'xX':
                    l = self.synth.render(lead_n, step_len * 2, "lead")
                    sl = min(len(l), len(master_bus) - cursor)
                    master_bus[cursor:cursor+sl] += l[:sl] * 0.5

                cursor += step_len

        # --- MASTERING ---
        print("/// MASTERING... ///")
        # Soft Clip (Tape Saturation Simulation)
        master_bus = np.tanh(master_bus * 1.5)
        # Normalize
        peak = np.max(np.abs(master_bus))
        if peak > 0: master_bus = (master_bus / peak) * 0.95
        
        fn = f"HIPHOP_FULL_{datetime.datetime.now().strftime('%H%M%S')}.wav"
        with wave.open(fn, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes((master_bus * 32767).astype(np.int16).tobytes())
        print(f"/// DONE: {fn} ///")

if __name__ == "__main__":
    p = SongArranger()
    p.render_full_song()
