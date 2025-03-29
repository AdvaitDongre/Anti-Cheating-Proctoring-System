import os
import pyaudio
import numpy as np
import wave
import time
import base64
import threading
import tempfile
import google.generativeai as genai
from dotenv import load_dotenv

class GeminiSpeakerDetector:
    def __init__(self, api_key, format=pyaudio.paInt16, channels=1, rate=16000, chunk=1024, 
                 record_seconds=3):
        self.format = format
        self.channels = channels
        self.rate = rate
        self.chunk = chunk
        self.record_seconds = record_seconds
        
        # Configure Gemini API
        os.environ["GEMINI_API_KEY"] = api_key
        genai.configure(api_key=api_key)
        
        # Create the model
        generation_config = {
            "temperature": 0.1,  # Lower temperature for more deterministic responses
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }
        
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
        )
        
        # Initialize chat session
        self.chat_session = self.model.start_chat(history=[])
        
        # PyAudio setup
        self.p = pyaudio.PyAudio()
        self.stream = None
        
        # Recording state
        self.is_recording = False
        self.stop_recording = False
        
        # Temporary file for audio
        self.temp_dir = tempfile.mkdtemp()
        self.temp_wav = os.path.join(self.temp_dir, "temp_audio.wav")
        
    def record_audio(self):
        """Record audio continuously and process in chunks"""
        self.is_recording = True
        self.stop_recording = False
        
        self.stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        
        print("Starting speaker detection...")
        print("Press Ctrl+C to stop")
        
        while not self.stop_recording:
            # Collect audio for specified duration
            frames = []
            for i in range(0, int(self.rate / self.chunk * self.record_seconds)):
                if self.stop_recording:
                    break
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                frames.append(data)
            
            if frames:  # Only process if we have audio data
                # Save the audio to a temporary WAV file
                self._save_audio(frames)
                
                # Analyze the audio for speaker count
                self._analyze_audio()
                
    def _save_audio(self, frames):
        """Save recorded audio to a temporary WAV file"""
        wf = wave.open(self.temp_wav, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.p.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(frames))
        wf.close()
    
    def _analyze_audio(self):
        """Send audio to Gemini API for speaker detection"""
        try:
            # Read the audio file
            with open(self.temp_wav, "rb") as audio_file:
                audio_data = audio_file.read()
            
            # Create prompt with audio data
            prompt = "Analyze this audio and tell me exactly how many distinct human speakers are in it. Only output a single number: 1 if there's only one speaker, or 2+ if there are multiple speakers. No explanation needed, just the number."
            
            # Send to Gemini with multimodal input
            response = self.chat_session.send_message([
                prompt,
                {
                    "inline_data": {
                        "mime_type": "audio/wav",
                        "data": base64.b64encode(audio_data).decode("utf-8")
                    }
                }
            ])
            
            # Extract the response text
            text_response = response.text.strip()
            
            # Check if response indicates multiple speakers
            if "2" in text_response or "multiple" in text_response.lower():
                print("\033[91m[ALERT] MULTIPLE SPEAKERS DETECTED!\033[0m")  # Red alert
            else:
                print("\033[92m[OK] Single speaker detected\033[0m")  # Green status
                
        except Exception as e:
            print(f"Error in audio analysis: {e}")
    
    def start(self):
        """Start the recording process in a separate thread"""
        self.recording_thread = threading.Thread(target=self.record_audio)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
        try:
            # Keep the main thread alive until Ctrl+C
            while self.recording_thread.is_alive():
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop()
            print("Speaker detection stopped")
    
    def stop(self):
        """Stop the recording process"""
        self.stop_recording = True
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.is_recording = False
        
        # Cleanup temporary files
        if os.path.exists(self.temp_wav):
            try:
                os.remove(self.temp_wav)
                os.rmdir(self.temp_dir)
            except:
                pass
    
    def close(self):
        """Clean up PyAudio"""
        self.stop()
        self.p.terminate()

if __name__ == "__main__":
    # Replace with your actual Gemini API key
    load_dotenv()
    
    API_KEY = os.environ["GEMINI_API_KEY"]
    
    # Initialize and start the detector
    detector = GeminiSpeakerDetector(api_key=API_KEY, record_seconds=3)
    try:
        detector.start()
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        detector.close()