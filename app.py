import numpy as np
import wave
import json
import collections
import secrets
import datetime
import random
from openai import OpenAI

# --- 1. GLOBAL STUDIO CONFIG ---
SAMPLE_RATE = 44100
BPM = 94
TOTAL_BARS = 36 # 1:30 song
LM_STUDIO_URL = "http://127.0.0.1:1234/v1"
BASE_STEP_SAMPLES = int((60.0 / BPM / 4.0) * SAMPLE_RATE)

# --- 2. TIMING: THE SWING ---
class PlatinumClock:
    def __init__(self):
        self.sys_random = secrets.SystemRandom()
        self.swing_amount = 0.22 

    def get_step_length(self, step_index):
        if step_index % 2 == 0:
            swing_offset = int(BASE_STEP_SAMPLES * self.swing_amount)
        else:
            swing_offset = -int(BASE_STEP_SAMPLES * self.swing_amount)
        jitter = self.sys_random.randint(-100, 100) 
        return BASE_STEP_SAMPLES + swing_offset + jitter

# --- 3. DRUMS: LAYERED KITS ---
class LayeredDrumModule:
    def __init__(self):
        print(">>> Loading Sample Packs...")
        self.kicks = [self._make_kick() for _ in range(5)]
        self.snares = [self._make_snare() for _ in range(5)]
        self.hats = [self._make_hat() for _ in range(4)]
        self.crash = self._make_crash() 

    def _make_kick(self):
        t = np.linspace(0, 0.5, int(SAMPLE_RATE * 0.5))
        freq = 110 * np.exp(-15 * t) + 30
        audio = np.sin(2 * np.pi * np.cumsum(freq) / SAMPLE_RATE)
        return np.tanh(audio * 3.0).astype(np.float32)

    def _make_snare(self):
        t = np.linspace(0, 0.25, int(SAMPLE_RATE * 0.25))
        noise = np.random.uniform(-1, 1, len(t)) * np.exp(-12 * t)
        tone = np.sin(2 * np.pi * 180 * t) * np.exp(-10 * t) * 0.3
        return ((noise + tone) * 0.8).astype(np.float32)

    def _make_hat(self):
        t = np.linspace(0, 0.05, int(SAMPLE_RATE * 0.05))
        return (np.random.uniform(-0.5, 0.5, len(t)) * np.exp(-30 * t)).astype(np.float32)

    def _make_crash(self):
        t = np.linspace(0, 2.0, int(SAMPLE_RATE * 2.0))
        noise = np.random.uniform(-0.4, 0.4, len(t))
        return (noise * np.exp(-2 * t)).astype(np.float32)

    def get_kick(self): return secrets.choice(self.kicks)
    def get_snare(self): return secrets.choice(self.snares)
    def get_hat(self): return secrets.choice(self.hats)

# --- 4. INSTRUMENTS: THE FULL BAND ---
class MultiSynth:
    def render(self, midi, duration, type="bass"):
        freq = 440.0 * (2.0**((midi - 69.0) / 12.0))
        t = np.arange(duration) / SAMPLE_RATE
        
        if type == "bass":
            osc1 = ((t * freq) % 1.0) * 2.0 - 1.0 
            osc2 = np.sin(2 * np.pi * (freq * 0.5) * t) 
            raw = osc1 * 0.6 + osc2 * 0.8 
            raw = np.convolve(raw, np.ones(10)/10, mode='same')
            return np.tanh(raw * 2.0) * 0.8

        elif type == "lead":
            mod = np.sin(2 * np.pi * 6.0 * t) * 0.005
            return np.sin(2 * np.pi * freq * (t + mod)) * 0.5

        elif type == "pad":
            osc1 = ((t * freq) % 1.0) * 2.0 - 1.0
            osc2 = ((t * freq * 1.01) % 1.0) * 2.0 - 1.0
            osc3 = ((t * freq * 0.99) % 1.0) * 2.0 - 1.0
            raw = (osc1 + osc2 + osc3) * 0.3
            att = int(0.3 * SAMPLE_RATE)
            if att < len(raw): raw[:att] *= np.linspace(0, 1, att)
            return raw * 0.3 

        elif type == "pluck":
            mod_freq = freq * 2.5
            mod = np.sin(2 * np.pi * mod_freq * t) * np.exp(-10 * t) * 2.0
            carrier = np.sin(2 * np.pi * freq * t + mod)
            return carrier * np.exp(-8 * t) * 0.4

        return np.zeros(duration)

# --- 5. THE PRODUCER (AI BRAIN) ---
class StudioProducer:
    def __init__(self):
        self.client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")
        self.clock = PlatinumClock()
        self.drums = LayeredDrumModule()
        self.synth = MultiSynth()
        self.motif_bank = {} 

    def get_structure(self, bar):
        if bar < 4: return "INTRO"
        if bar < 20: return "VERSE"
        if bar < 28: return "HOOK"
        return "OUTRO"

    def get_pattern(self, bar):
        section = self.get_structure(bar)
        
        if section == "HOOK" and "hook_pat" in self.motif_bank:
            return self.motif_bank["hook_pat"]

        prompt = (
            f"Role: Hip Hop Producer. Section: {section}. BPM: {BPM}.\n"
            "Output JSON ONLY. Create a 1-bar loop.\n"
            "Format: {\"root\":\"C2\", \"kick\":\"x...\", \"snare\":\"...\", \"bass\":\"x...\", \"pad_chord\":\"C3\", \"pluck_pat\":\"--x-\", \"lead_pat\":\"...\"}"
        )

        try:
            print(f"-> Composing Bar {bar+1} ({section})...")
            resp = self.client.chat.completions.create(
                model="model-identifier",
                messages=[{"role": "system", "content": prompt}],
                temperature=0.9, max_tokens=400, stop=["}"]
            )
            raw = resp.choices[0].message.content
            if "}" not in raw: raw += "}"
            data = json.loads(raw[raw.find('{'):raw.rfind('}')+1])
            
            if section == "HOOK" and "hook_pat" not in self.motif_bank:
                self.motif_bank["hook_pat"] = data
            return data
        except:
            return {"root":"C2", "kick":"x---", "snare":"----", "bass":"x---"}

    def render(self):
        print(f"/// RECORDING STUDIO SESSION ({TOTAL_BARS} BARS) ///")
        full_len = int(TOTAL_BARS * 16 * BASE_STEP_SAMPLES * 1.05)
        track_mix = np.zeros(full_len)
        cursor = 0
        
        note_map = {'C2':36, 'F2':41, 'G2':43, 'C3':48, 'G3':55, 'C4':60, 'G4':67, 'C5':72}

        for bar in range(TOTAL_BARS):
            data = self.get_pattern(bar)
            
            # --- ERROR FIX: HANDLE CHORDS AS LISTS ---
            # If AI returns ["C3", "E3", "G3"], just take "C3"
            raw_root = data.get('root', 'C2')
            if isinstance(raw_root, list): raw_root = raw_root[0]
            
            raw_pad = data.get('pad_chord', 'C3')
            if isinstance(raw_pad, list): raw_pad = raw_pad[0]
            
            root = note_map.get(raw_root, 36)
            pad_n = note_map.get(raw_pad, 48)
            # -----------------------------------------
            
            # Crash Logic
            if bar in [0, 4, 20, 28]:
                c = self.drums.crash
                cl = min(len(c), len(track_mix) - cursor)
                track_mix[cursor:cursor+cl] += c[:cl] * 0.4

            for step in range(16):
                step_len = self.clock.get_step_length(step)
                
                # DRUMS
                if step < len(data.get('kick','')) and data['kick'][step] in 'xX':
                    s = self.drums.get_kick()
                    sl = min(len(s), len(track_mix) - cursor)
                    track_mix[cursor:cursor+sl] += s[:sl]
                
                if step < len(data.get('snare','')) and data['snare'][step] in 'xX':
                    s = self.drums.get_snare()
                    sl = min(len(s), len(track_mix) - cursor)
                    track_mix[cursor:cursor+sl] += s[:sl]
                
                if step % 2 == 0: 
                    h = self.drums.get_hat()
                    hl = min(len(h), len(track_mix) - cursor)
                    track_mix[cursor:cursor+hl] += h[:hl] * 0.3

                # SYNTHS
                if step < len(data.get('bass','')) and data['bass'][step] in 'xX':
                    b = self.synth.render(root, step_len * 2, "bass")
                    sl = min(len(b), len(track_mix) - cursor)
                    track_mix[cursor:cursor+sl] += b[:sl]

                if step == 0: # Pad once per bar
                    p = self.synth.render(pad_n, step_len * 16, "pad")
                    sl = min(len(p), len(track_mix) - cursor)
                    track_mix[cursor:cursor+sl] += p[:sl]

                if step < len(data.get('lead_pat','')) and data['lead_pat'][step] in 'xX':
                    l = self.synth.render(root + 24, step_len * 2, "lead") 
                    sl = min(len(l), len(track_mix) - cursor)
                    track_mix[cursor:cursor+sl] += l[:sl]

                if step < len(data.get('pluck_pat','')) and data['pluck_pat'][step] in 'xX':
                    pl = self.synth.render(root + 36, step_len, "pluck") 
                    sl = min(len(pl), len(track_mix) - cursor)
                    track_mix[cursor:cursor+sl] += pl[:sl]

                cursor += step_len

        print("/// MASTERING... ///")
        track_mix = np.tanh(track_mix * 1.2) 
        mx = np.max(np.abs(track_mix))
        if mx > 0: track_mix = (track_mix / mx) * 0.95
        
        fn = f"PLATINUM_HIT_{datetime.datetime.now().strftime('%H%M%S')}.wav"
        with wave.open(fn, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes((track_mix * 32767).astype(np.int16).tobytes())
        print(f"/// DONE: {fn} ///")

if __name__ == "__main__":
    p = StudioProducer()
    p.render()
    
