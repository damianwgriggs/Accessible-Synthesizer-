import numpy as np
import wave
import json
import collections
import secrets
import datetime
import random
from openai import OpenAI

# --- 1. STUDIO CONFIG ---
SAMPLE_RATE = 44100
BPM = 86
TOTAL_BARS = 48 
LM_STUDIO_URL = "http://127.0.0.1:1234/v1"
BASE_STEP_SAMPLES = int((60.0 / BPM / 4.0) * SAMPLE_RATE)

# --- 2. THE HUMANIZER (NEW) ---
class PersonalityEngine:
    def __init__(self):
        # Offsets in milliseconds (Negative = Rush, Positive = Drag)
        self.offsets = {
            'kick': 0,        # The Anchor
            'snare': 15,      # Lazy, dragging snare (Dilla style)
            'hat': -5,        # Rushing hats (Urgency)
            'bass': 20,       # Bassist playing waaaay back in the pocket
            'keys': -10,      # Eager keys
            'lead': 5
        }
    
    def get_start_index(self, cursor, inst_type):
        # Convert ms to samples
        ms_offset = self.offsets.get(inst_type, 0)
        # Add random "human error" per note (+/- 4ms)
        human_error = random.randint(-176, 176) 
        total_offset_samples = int((ms_offset / 1000.0) * SAMPLE_RATE) + human_error
        
        return cursor + total_offset_samples

# --- 3. CLOCK ---
class JamClock:
    def __init__(self):
        self.sys_random = secrets.SystemRandom()
        self.swing_amount = 0.28 

    def get_step_length(self, step_index):
        if step_index % 2 == 0:
            swing_offset = int(BASE_STEP_SAMPLES * self.swing_amount)
        else:
            swing_offset = -int(BASE_STEP_SAMPLES * self.swing_amount)
        jitter = self.sys_random.randint(-100, 100) 
        return BASE_STEP_SAMPLES + swing_offset + jitter

# --- 4. DRUMS ---
class StonerDrums:
    def __init__(self):
        print(">>> Tuning Skins...")
        self.kicks = [self._make_kick() for _ in range(5)]
        self.snares = [self._make_snare() for _ in range(5)]
        self.hats = [self._make_hat() for _ in range(4)]

    def _make_kick(self):
        t = np.linspace(0, 0.4, int(SAMPLE_RATE * 0.4))
        freq = 90 * np.exp(-10 * t) + 40
        audio = np.sin(2 * np.pi * np.cumsum(freq) / SAMPLE_RATE)
        return (np.tanh(audio * 1.5) * np.exp(-4 * t)).astype(np.float32)

    def _make_snare(self):
        t = np.linspace(0, 0.15, int(SAMPLE_RATE * 0.15))
        tone = np.sin(2 * np.pi * 350 * t) * np.exp(-15 * t)
        noise = np.random.uniform(-0.8, 0.8, len(t)) * np.exp(-20 * t)
        return ((tone + noise) * 0.7).astype(np.float32)

    def _make_hat(self):
        t = np.linspace(0, 0.06, int(SAMPLE_RATE * 0.06))
        return (np.random.uniform(-0.4, 0.4, len(t)) * np.exp(-15 * t)).astype(np.float32)

    def get_kick(self): return secrets.choice(self.kicks)
    def get_snare(self): return secrets.choice(self.snares)
    def get_hat(self): return secrets.choice(self.hats)

# --- 5. SYNTHS ---
class JamSynth:
    def render(self, midi, duration, type="bass"):
        freq = 440.0 * (2.0**((midi - 69.0) / 12.0))
        # RANDOMIZE DURATION (Staccato vs Legato)
        # Musicians don't hold notes for exact grid time
        human_duration = int(duration * random.uniform(0.8, 1.1))
        t = np.arange(human_duration) / SAMPLE_RATE
        
        detune = random.uniform(0.995, 1.005)
        
        if type == "bass":
            osc = ((t * freq * detune) % 1.0) * 2.0 - 1.0
            env = np.exp(-8 * t)
            osc = np.convolve(osc, np.ones(5)/5, mode='same')
            return osc * env * 0.8

        elif type == "keys":
            raw = np.sin(2 * np.pi * freq * t)
            lfo = np.sin(2 * np.pi * 3.0 * t)
            return raw * (0.5 + 0.5 * lfo) * np.exp(-2 * t) * 0.5

        elif type == "lead":
            glide = np.sin(2 * np.pi * 4.0 * t) * 0.005
            return np.sin(2 * np.pi * freq * (t + glide)) * 0.4

        return np.zeros(human_duration)

# --- 6. PRODUCER ---
class InfiniteJamProducer:
    def __init__(self):
        self.client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")
        self.clock = JamClock()
        self.personality = PersonalityEngine() # NEW
        self.drums = StonerDrums()
        self.synth = JamSynth()
        self.history = collections.deque(maxlen=7) 

    def get_next_bar_improv(self, bar_num):
        context = "Start."
        if self.history:
            context = "->".join([h.get('root', 'C2') for h in self.history])

        prompt = (
            f"Genre: Stoner Funk. BPM: {BPM}. Bar: {bar_num}.\n"
            "Output JSON ONLY. IMPROVISE.\n"
            f"Context: {context}\n"
            "Format: {\"root\":\"C2\", \"kick\":\"x...\", \"snare\":\"...\", \"bass\":\"x...\", \"keys_chord\":\"C3\", \"lead_pat\":\"...\"}"
        )

        try:
            print(f"-> Improvising Bar {bar_num+1}...")
            resp = self.client.chat.completions.create(
                model="model-identifier",
                messages=[{"role": "system", "content": prompt}],
                temperature=1.1, 
                max_tokens=400, stop=["}"]
            )
            raw = resp.choices[0].message.content
            if "}" not in raw: raw += "}"
            data = json.loads(raw[raw.find('{'):raw.rfind('}')+1])
            self.history.append(data)
            return data
        except:
            return {"root":"C2", "kick":"x---x---", "snare":"----x---", "bass":"x-x-"}

    def add_to_mix(self, mix, audio, start_idx):
        """Safely add audio to mix handling array bounds"""
        if start_idx < 0: start_idx = 0
        if start_idx >= len(mix): return
        
        end_idx = start_idx + len(audio)
        if end_idx > len(mix):
            audio = audio[:len(mix)-start_idx]
            end_idx = len(mix)
            
        mix[start_idx:end_idx] += audio

    def render(self):
        print(f"/// DE-QUANTIZED SESSION ({TOTAL_BARS} BARS) ///")
        # Add extra buffer for delays/reverbs tail
        full_len = int(TOTAL_BARS * 16 * BASE_STEP_SAMPLES * 1.1)
        track_mix = np.zeros(full_len)
        cursor = 0
        
        note_map = {'C2':36, 'Eb2':39, 'F2':41, 'G2':43, 'Bb2':46, 'C3':48, 'Eb3':51, 'G3':55, 'C4':60, 'G4':67, 'C5':72}

        for bar in range(TOTAL_BARS):
            data = self.get_next_bar_improv(bar)
            
            r_root = data.get('root', 'C2')
            if isinstance(r_root, list): r_root = r_root[0]
            r_keys = data.get('keys_chord', 'C3')
            if isinstance(r_keys, list): r_keys = r_keys[0]
            
            root = note_map.get(r_root, 36)
            keys_n = note_map.get(r_keys, 48)

            for step in range(16):
                step_len = self.clock.get_step_length(step)
                
                # --- DRUMS (OFFSET APPLIED) ---
                vel = random.uniform(0.8, 1.0)
                
                if step < len(data.get('kick','')) and data['kick'][step] in 'xX':
                    s = self.drums.get_kick()
                    # Personality: Kick is the anchor (0ms), but has human error
                    start = self.personality.get_start_index(cursor, 'kick')
                    self.add_to_mix(track_mix, s * vel, start)

                if step < len(data.get('snare','')) and data['snare'][step] in 'xX':
                    s = self.drums.get_snare()
                    # Personality: Snare is lazy (+15ms)
                    start = self.personality.get_start_index(cursor, 'snare')
                    self.add_to_mix(track_mix, s * vel, start)

                if random.random() > 0.15: 
                    h = self.drums.get_hat()
                    # Personality: Hats are rushing (-5ms)
                    start = self.personality.get_start_index(cursor, 'hat')
                    self.add_to_mix(track_mix, h * 0.3, start)

                # --- SYNTHS (OFFSET APPLIED) ---
                if step < len(data.get('bass','')) and data['bass'][step] in 'xX':
                    b = self.synth.render(root, step_len * 2, "bass")
                    # Personality: Bass drags huge (+20ms)
                    start = self.personality.get_start_index(cursor, 'bass')
                    self.add_to_mix(track_mix, b, start)

                if step < len(data.get('keys_pat','')) and data['keys_pat'][step] in 'xX':
                     k = self.synth.render(keys_n, step_len * 4, "keys")
                     # Personality: Keys rush (-10ms)
                     start = self.personality.get_start_index(cursor, 'keys')
                     self.add_to_mix(track_mix, k, start)
                elif step == 0 and random.random() > 0.4:
                     k = self.synth.render(keys_n, step_len * 8, "keys")
                     start = self.personality.get_start_index(cursor, 'keys')
                     self.add_to_mix(track_mix, k, start)

                if step < len(data.get('lead_pat','')) and data['lead_pat'][step] in 'xX':
                    l = self.synth.render(root + 24, step_len * 2, "lead") 
                    start = self.personality.get_start_index(cursor, 'lead')
                    self.add_to_mix(track_mix, l, start)

                cursor += step_len

        print("/// MASTERING... ///")
        track_mix = np.tanh(track_mix * 1.3) 
        mx = np.max(np.abs(track_mix))
        if mx > 0: track_mix = (track_mix / mx) * 0.95
        
        fn = f"ORGANIC_JAM_{datetime.datetime.now().strftime('%H%M%S')}.wav"
        with wave.open(fn, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes((track_mix * 32767).astype(np.int16).tobytes())
        print(f"/// DONE: {fn} ///")

if __name__ == "__main__":
    p = InfiniteJamProducer()
    p.render()
