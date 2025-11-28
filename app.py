import tkinter as tk
from tkinter import ttk, Scale, OptionMenu, filedialog, messagebox
import numpy as np
import pyaudio
import threading
import wave  # ### NEW: Required for saving WAV files
import struct # ### NEW: Required for binary data packing

# --- 1. GLOBAL CONSTANTS ---
SAMPLE_RATE = 44100
BLOCK_SIZE = 512
MAX_GAIN = 0.2
KEY_MAP = {
    'a': 60, 'w': 61, 's': 62, 'e': 63, 'd': 64, 'f': 65,
    't': 66, 'g': 67, 'y': 68, 'h': 69, 'u': 70, 'j': 71, 
    'k': 72, 
}

# --- 2. SYNTH COMPONENT CLASSES (VCO, ADSR, VCF) ---
class VCO:
    def __init__(self, sample_rate=SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.waveform = 'sine'
        self.frequency = 440.0
        self._phase = 0.0
        self.coarse_tune = 0
        self.fine_tune = 0.0

    def set_midi_note(self, midi_note):
        tuned_note = midi_note + self.coarse_tune + self.fine_tune
        self.frequency = 440.0 * (2.0**((tuned_note - 69.0) / 12.0))
        
    def generate_block(self, num_samples=BLOCK_SIZE):
        time_vector = np.arange(num_samples) / self.sample_rate
        phase_increment = 2.0 * np.pi * self.frequency * time_vector
        
        phase_total = phase_increment + self._phase
        self._phase = phase_total[-1] % (2.0 * np.pi)
        
        if self.waveform == 'sine':
            audio_block = np.sin(phase_total)
        elif self.waveform == 'square':
            audio_block = np.sign(np.sin(phase_total))
        elif self.waveform == 'sawtooth':
            audio_block = ((phase_total % (2.0 * np.pi)) / (2.0 * np.pi)) * 2.0 - 1.0
        elif self.waveform == 'triangle':
            audio_block = (2/np.pi) * np.arcsin(np.sin(phase_total))
        else:
            audio_block = np.sin(phase_total)
            
        return audio_block

class ADSR:
    def __init__(self, sample_rate=SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.A, self.D, self.S, self.R = 0.1, 0.1, 0.7, 0.5 
        
        self.state = 'idle'
        self._level = 0.0
        self._time = 0.0
        self._sustain_level = 0.0

    def note_on(self, global_params):
        self.A = global_params.get('attack', 0.1)
        self.D = global_params.get('decay', 0.1)
        self.S = global_params.get('sustain', 0.7)
        self.R = global_params.get('release', 0.5)
        
        self.state = 'attack'
        self._time = 0.0
        
    def note_off(self):
        if self.state != 'idle':
            self.state = 'release'
            self._time = 0.0

    def process_block(self, num_samples=BLOCK_SIZE):
        gain_block = np.zeros(num_samples)
        dt = 1.0 / self.sample_rate
        
        current_level = self._level
        
        for i in range(num_samples):
            if self.state == 'attack':
                if self.A > 0:
                    current_level += dt / self.A
                if current_level >= 1.0:
                    current_level = 1.0
                    self.state = 'decay'
                    self._time = 0.0
            
            elif self.state == 'decay':
                if self.D > 0 and current_level > self.S:
                    current_level -= ((1.0 - self.S) * dt) / self.D
                if current_level <= self.S:
                    current_level = self.S
                    self.state = 'sustain'

            elif self.state == 'sustain':
                current_level = self.S

            elif self.state == 'release':
                if self.R > 0:
                    current_level -= (self._sustain_level * dt) / self.R
                if current_level <= 0.0:
                    current_level = 0.0
                    self.state = 'idle'
            
            gain_block[i] = current_level
            self._time += dt
            self._level = current_level
            
            if self.state != 'release':
                 self._sustain_level = current_level 

        return gain_block

class VCF:
    def __init__(self, sample_rate=SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.cutoff = 20000.0
        self.resonance = 0.0
        self._y1 = 0.0

    def process_block(self, audio_in):
        tau = 1.0 / (2.0 * np.pi * self.cutoff)
        alpha = (self.sample_rate * tau) / (1.0 + self.sample_rate * tau)
        
        audio_out = np.zeros_like(audio_in)
        
        for i in range(len(audio_in)):
            audio_out[i] = alpha * audio_in[i] + (1.0 - alpha) * self._y1
            self._y1 = audio_out[i]
            
        return audio_out

class Voice:
    def __init__(self, midi_note, synth_settings):
        self.vco = VCO(SAMPLE_RATE)
        self.amp_env = ADSR(SAMPLE_RATE)
        self.vco.set_midi_note(midi_note)
        self.amp_env.note_on(synth_settings)

    def generate_block(self, num_samples):
        osc_out = self.vco.generate_block(num_samples)
        amp_gain = self.amp_env.process_block(num_samples)
        
        is_finished = (self.amp_env.state == 'idle')
        
        return osc_out * amp_gain, is_finished

# --- 3. MAIN SYNTHESIZER ENGINE ---
class ClaritySynthesizer:
    def __init__(self):
        
        self.MAX_POLYPHONY = 8
        self.voice_lock = threading.Lock()
        
        self.active_voices = {}
        self.global_vcf = VCF(SAMPLE_RATE)
        self.is_running = True
        
        # ### NEW: Recording State
        self.is_recording = False
        self.record_buffer = [] # Stores numpy blocks
        
        self.params = {
            'master_volume': MAX_GAIN,
            'waveform': 'sine',
            'coarse_tune': 0,
            'fine_tune': 0.0,
            'cutoff': 10000.0,
            'resonance': 0.0,
            'attack': 0.1,
            'decay': 0.1,
            'sustain': 0.7,
            'release': 0.5,
        }
        
        p = pyaudio.PyAudio()
        self.stream = p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=SAMPLE_RATE,
            output=True,
            frames_per_buffer=BLOCK_SIZE,
            stream_callback=self._audio_callback
        )
        print("Clarity Synthesizer Audio Stream Active.")

    def note_on(self, midi_note):
        with self.voice_lock: 
            if midi_note not in self.active_voices:
                if len(self.active_voices) >= self.MAX_POLYPHONY:
                    return

                new_voice = Voice(midi_note, self.params)
                new_voice.vco.waveform = self.params['waveform']
                new_voice.vco.coarse_tune = self.params['coarse_tune']
                new_voice.vco.fine_tune = self.params['fine_tune']
                self.active_voices[midi_note] = new_voice

    def note_off(self, midi_note):
        with self.voice_lock:
            if midi_note in self.active_voices:
                self.active_voices[midi_note].amp_env.note_off()

    def update_param(self, name, value):
        if name in ['coarse_tune']:
            value = int(value)
        elif name in ['cutoff']:
            value = float(value)
        
        self.params[name] = value
        
        if name == 'cutoff':
            self.global_vcf.cutoff = value
        
        if name in ['waveform', 'coarse_tune', 'fine_tune']:
            for midi_note, voice in self.active_voices.items():
                setattr(voice.vco, name, value)
                if name in ['coarse_tune', 'fine_tune']:
                     voice.vco.set_midi_note(midi_note)

    # ### NEW: Recording Controls
    def start_recording(self):
        self.record_buffer = []
        self.is_recording = True
        print("Recording Started...")

    def stop_recording(self):
        self.is_recording = False
        print(f"Recording Stopped. Captured {len(self.record_buffer)} blocks.")
    
    def save_recording(self, filename):
        if not self.record_buffer:
            return False
            
        try:
            # Combine all numpy blocks into one large array
            full_audio = np.concatenate(self.record_buffer)
            
            # Convert 32-bit float (-1.0 to 1.0) to 16-bit PCM integer (-32767 to 32767)
            # This ensures compatibility with all standard media players
            audio_int16 = (full_audio * 32767).clip(-32767, 32767).astype(np.int16)
            
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(1) # Mono
                wf.setsampwidth(2) # 2 bytes = 16 bit
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio_int16.tobytes())
            return True
        except Exception as e:
            print(f"Error saving file: {e}")
            return False

    def _audio_callback(self, in_data, frame_count, time_info, status):
        output_block = np.zeros(frame_count, dtype=np.float32)
        voices_to_remove = []
        
        with self.voice_lock:
            active_items = list(self.active_voices.items())
        
        for midi_note, voice in active_items:
            voice_output, is_finished = voice.generate_block(frame_count)
            output_block += voice_output
            
            if is_finished:
                voices_to_remove.append(midi_note)

        with self.voice_lock:
            for midi_note in voices_to_remove:
                if midi_note in self.active_voices:
                    del self.active_voices[midi_note]
        
        output_block = self.global_vcf.process_block(output_block)

        master_gain = self.params.get('master_volume', MAX_GAIN)
        output_block = np.clip(output_block, -1.0, 1.0) * master_gain
        
        # ### NEW: Capture audio if recording
        # We capture the float array before conversion to bytes for processing
        if self.is_recording:
            self.record_buffer.append(output_block.copy())

        return (output_block.astype(np.float32).tobytes(), pyaudio.paContinue)

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        p = pyaudio.PyAudio()
        p.terminate()

# --- 4. ACCESSIBLE TKINTER UI (SCROLL FUNCTION ADDED) ---

class ClarityUI(tk.Tk):
    """The main tkinter application with VI-First accessibility."""
    
    def __init__(self, synth):
        super().__init__()
        self.synth = synth
        self.title("Clarity VI-First Synthesizer")
        
        # Set large window size
        self.geometry("900x1400") 
        
        # --- High-Contrast & Font Setup ---
        self.configure(bg='black')
        LARGE_FONT = ('Arial', 28)
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TLabel', background='black', foreground='white', font=LARGE_FONT)
        style.configure('TFrame', background='black')
        style.configure('TScale', background='black', foreground='white', troughcolor='#333333')
        style.configure('TButton', background='#333333', foreground='white', font=LARGE_FONT, borderwidth=5)
        style.configure('TMenubutton', background='black', foreground='white', font=LARGE_FONT)

        style.configure('Focused.TScale', background='black', foreground='white', troughcolor='yellow')
        
        # Styles for Record Button
        style.configure('Record.TButton', foreground='red', background='#220000')
        style.configure('Stop.TButton', foreground='yellow', background='#222200')

        # ------------------------------------------------------------------
        # SCROLLING IMPLEMENTATION
        # ------------------------------------------------------------------
        
        canvas = tk.Canvas(self, bg='black', highlightthickness=0)
        canvas.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        scrollbar.pack(side="right", fill="y")

        canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollable_frame = ttk.Frame(canvas, padding="30 30 30 30")
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )
        scrollable_frame.bind("<MouseWheel>", self._on_mousewheel) 
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        
        scrollable_frame.bind(
            "<Enter>",
            lambda e: canvas.bind_all('<Key-Down>', self._on_arrow_key_scroll),
        )
        scrollable_frame.bind(
            "<Leave>",
            lambda e: canvas.unbind_all('<Key-Down>'),
        )

        # --- Keyboard Bindings ---
        self.bind_all('<KeyPress>', self._handle_key_press)
        self.bind_all('<KeyRelease>', self._handle_key_release)
        self.pressed_keys = set()

        # --- Build Layout ---
        self._row_counter = 0

        # Master Section
        self._add_section_header(scrollable_frame, "Master Controls")
        
        # ### NEW: Add Record Button in Master Section
        self._add_record_controls(scrollable_frame)
        
        self._add_scale_control(scrollable_frame, "master_volume", "Volume", 0.0, MAX_GAIN, MAX_GAIN, 0.01)

        # VCO Section
        self._add_section_header(scrollable_frame, "VCO (Oscillator)")
        self._add_dropdown_control(scrollable_frame, "waveform", "Waveform", ['sine', 'square', 'sawtooth', 'triangle'])
        self._add_scale_control(scrollable_frame, "coarse_tune", "Coarse Tune (Semi)", -12, 12, 0, 1)
        self._add_scale_control(scrollable_frame, "fine_tune", "Fine Tune (Cents)", -1.0, 1.0, 0.0, 0.01)

        # VCF Section
        self._add_section_header(scrollable_frame, "VCF (Filter)")
        self._add_scale_control(scrollable_frame, "cutoff", "Cutoff Freq (Hz)", 20, 20000, 10000, 1)
        self._add_scale_control(scrollable_frame, "resonance", "Resonance (Q)", 0.0, 1.0, 0.0, 0.01, state=tk.DISABLED)

        # ADSR Section
        self._add_section_header(scrollable_frame, "ADSR (Envelope)")
        self._add_scale_control(scrollable_frame, "attack", "Attack (s)", 0.001, 5.0, 0.1, 0.01)
        self._add_scale_control(scrollable_frame, "decay", "Decay (s)", 0.001, 5.0, 0.1, 0.01)
        self._add_scale_control(scrollable_frame, "sustain", "Sustain (Level)", 0.0, 1.0, 0.7, 0.01)
        self._add_scale_control(scrollable_frame, "release", "Release (s)", 0.001, 5.0, 0.5, 0.01)
        
        # --- Final Instructions Label ---
        ttk.Label(scrollable_frame, text="Note Keys: A S D F G... | Navigation: TAB/Arrows", 
                  background='#0000FF', foreground='white', font=('Arial', 28, 'bold')).grid(
                      row=self._row_counter, column=0, columnspan=2, pady=30, sticky='ew')
        self._row_counter += 1
        
        self.canvas = canvas 

    # ### NEW: Record Button Helper
    def _add_record_controls(self, parent):
        self.record_btn_var = tk.StringVar(value="RECORD")
        
        self.record_btn = ttk.Button(
            parent, 
            textvariable=self.record_btn_var, 
            command=self._toggle_record,
            style='Record.TButton'
        )
        self.record_btn.grid(row=self._row_counter, column=0, columnspan=2, sticky='ew', padx=15, pady=20)
        self._row_counter += 1

    # ### NEW: Record Toggle Logic
    def _toggle_record(self):
        if not self.synth.is_recording:
            # Start Recording
            self.synth.start_recording()
            self.record_btn_var.set("STOP & SAVE")
            self.record_btn.configure(style='Stop.TButton')
            print("Announcement: Recording started.")
        else:
            # Stop Recording
            self.synth.stop_recording()
            self.record_btn_var.set("RECORD")
            self.record_btn.configure(style='Record.TButton')
            
            # Open File Dialog
            filename = filedialog.asksaveasfilename(
                defaultextension=".wav",
                filetypes=[("WAV files", "*.wav")],
                title="Save Recording"
            )
            
            if filename:
                success = self.synth.save_recording(filename)
                if success:
                    print(f"Announcement: Saved to {filename}")
                else:
                    print("Announcement: Error saving file.")
            else:
                print("Announcement: Save cancelled.")

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
    def _on_arrow_key_scroll(self, event):
        if self.focus_get() == self.canvas:
             self.canvas.yview_scroll(1, "units")

    def _add_section_header(self, parent, text):
        ttk.Label(parent, text=f"--- {text} ---", foreground='yellow', font=('Arial', 32, 'bold')).grid(
            row=self._row_counter, column=0, columnspan=2, pady=(35, 15), sticky='w')
        self._row_counter += 1

    def _add_scale_control(self, parent, param_name, label_text, from_val, to_val, default_val, resolution, state=tk.NORMAL):
        label = ttk.Label(parent, text=label_text + ":", anchor='w')
        label.grid(row=self._row_counter, column=0, sticky='w', padx=15, pady=8)

        current_value = tk.DoubleVar(value=default_val)
        
        def update_synth(value):
            self.synth.update_param(param_name, float(value))
            print(f"Announcement: {label_text}, Value: {float(value):.2f}") 

        scale = ttk.Scale(parent, from_=from_val, to=to_val, orient=tk.HORIZONTAL, 
                          command=update_synth, variable=current_value, 
                          length=550, style='TScale', state=state, 
                          value=default_val, 
                          takefocus=1, 
                          cursor='hand2',
                          )
        scale.grid(row=self._row_counter, column=1, sticky='ew', padx=15, pady=8)
        
        scale.bind('<FocusIn>', lambda event: scale.configure(style='Focused.TScale'))
        scale.bind('<FocusOut>', lambda event: scale.configure(style='TScale'))

        self.synth.update_param(param_name, default_val)
        self._row_counter += 1

    def _add_dropdown_control(self, parent, param_name, label_text, options):
        label = ttk.Label(parent, text=label_text + ":", anchor='w')
        label.grid(row=self._row_counter, column=0, sticky='w', padx=15, pady=8)
        
        default_val = options[0]
        current_value = tk.StringVar(value=default_val)
        
        def update_synth(value):
            self.synth.update_param(param_name, value)
            print(f"Announcement: {label_text}, Set to: {value}")

        menu = OptionMenu(parent, current_value, default_val, *options, command=update_synth)
        menu.configure(bg='black', fg='white', font=('Arial', 28), takefocus=1)
        menu.grid(row=self._row_counter, column=1, sticky='ew', padx=15, pady=8)

        self.synth.update_param(param_name, default_val)
        self._row_counter += 1

    def _handle_key_press(self, event):
        key = event.keysym.lower()
        if key in KEY_MAP:
            midi_note = KEY_MAP[key]
            if key not in self.pressed_keys:
                self.synth.note_on(midi_note)
                print(f"Note On: {key} (MIDI {midi_note})")
                self.pressed_keys.add(key)
        
        if event.keysym == 'Return' and self.focus_get():
            try:
                widget = self.focus_get()
                if isinstance(widget, ttk.Scale):
                    widget_info = widget.grid_info()
                    parent = widget.master
                    label_text = parent.grid_slaves(row=widget_info['row'], column=0)[0].cget('text').replace(':', '')
                    print(f"Screen Read: {label_text} slider focused. Current Value: {widget.get():.2f}")
                elif isinstance(widget, OptionMenu):
                    print(f"Screen Read: Dropdown focused. Current Selection: {widget.cget('text')}")
            except Exception:
                pass

    def _handle_key_release(self, event):
        key = event.keysym.lower()
        if key in KEY_MAP:
            midi_note = KEY_MAP[key]
            self.synth.note_off(midi_note)
            print(f"Note Off: {key} (MIDI {midi_note})")
            if key in self.pressed_keys:
                self.pressed_keys.remove(key)

# --- 5. MAIN EXECUTION ---
if __name__ == '__main__':
    synth = ClaritySynthesizer()
    app = ClarityUI(synth)
    
    try:
        app.mainloop()
    except tk.TclError as e:
        print(f"Tkinter error during mainloop: {e}")
    finally:
        synth.close()
        print("\nClarity Synthesizer shut down successfully.")
