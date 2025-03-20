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

class SpeakerDetector:
    def __init__(self, api_key, format=pyaudio.paInt16, channels=1, rate=16000, chunk=1024, 
                 record_seconds=3):
        self.format = format
        self.channels = channels
        self.rate = rate
        self.chunk = chunk
        self.record_seconds = record_seconds
        load_dotenv()
        # Configure Gemini API
        api_key = os.getenv("GEMINI_API_KEY")
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
        
        # Statistics
        self.total_frames = 0
        self.multiple_speaker_frames = 0
        self.detection_start_time = None
        
    def record_audio(self):
        """Record audio continuously and process in chunks"""
        self.is_recording = True
        self.stop_recording = False
        self.detection_start_time = time.time()
        
        self.stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        
        print("Starting speaker detection...")
        
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
                result = self._analyze_audio()
                self.total_frames += 1
                if result:
                    self.multiple_speaker_frames += 1
                
    def _save_audio(self, frames):
        """Save recorded audio to a temporary WAV file"""
        wf = wave.open(self.temp_wav, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.p.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(frames))
        wf.close()
    
    def _analyze_audio(self):
        """Send audio to Gemini API for speaker detection
        Returns True if multiple speakers detected, False otherwise"""
        multiple_speakers_detected = False
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
                multiple_speakers_detected = True
                
        except Exception as e:
            print(f"Error in audio analysis: {e}")
            
        return multiple_speakers_detected
    
    def start(self):
        """Start the recording process in a separate thread"""
        self.recording_thread = threading.Thread(target=self.record_audio)
        self.recording_thread.daemon = True
        self.recording_thread.start()
    
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
        
    def get_statistics(self):
        """Return statistics about speaker detection"""
        elapsed_time = 0
        if self.detection_start_time:
            elapsed_time = time.time() - self.detection_start_time
            
        stats = {
            "total_frames": self.total_frames,
            "multiple_speaker_frames": self.multiple_speaker_frames,
            "detection_percentage": (self.multiple_speaker_frames / self.total_frames * 100) if self.total_frames > 0 else 0,
            "elapsed_time": elapsed_time
        }
        return stats
    
    def multiple_speakers_detected(self):
        """Check if multiple speakers are currently detected"""
        # Consider multiple speakers detected if at least 2 of the last 3 frames had multiple speakers
        # This is a simple heuristic to reduce false positives
        return self.multiple_speaker_frames > 0 and self.total_frames > 0
    


def main():
    import os
    import time
    import sys
    import signal
    from dotenv import load_dotenv
    
    # Load environment variables for API key
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("Error: GEMINI_API_KEY not found in environment variables.")
        print("Please create a .env file with your GEMINI_API_KEY=your_key_here")
        sys.exit(1)
    
    # Create the speaker detector
    detector = SpeakerDetector(api_key=api_key)
    
    # Handle graceful shutdown
    def signal_handler(sig, frame):
        print("\nShutting down speaker detection...")
        detector.stop()
        detector.close()
        
        # Display final statistics
        stats = detector.get_statistics()
        print("\nFinal Statistics:")
        print(f"Total audio chunks analyzed: {stats['total_frames']}")
        print(f"Chunks with multiple speakers detected: {stats['multiple_speaker_frames']}")
        print(f"Multiple speaker detection percentage: {stats['detection_percentage']:.2f}%")
        print(f"Total elapsed time: {stats['elapsed_time']:.2f} seconds")
        
        sys.exit(0)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start the detection
    print("Starting speaker detection system...")
    print("Press Ctrl+C to stop")
    
    # Start recording and detection
    detector.start()
    
    try:
        # Display real-time statistics while running
        last_status_time = time.time()
        status_interval = 2  # Update status every 2 seconds
        
        while True:
            current_time = time.time()
            
            # Print status update at regular intervals
            if current_time - last_status_time >= status_interval:
                stats = detector.get_statistics()
                
                # Clear line and print status
                sys.stdout.write("\r" + " " * 80 + "\r")  # Clear line
                sys.stdout.write(f"Analyzed: {stats['total_frames']} chunks | ")
                sys.stdout.write(f"Multiple speakers: {stats['multiple_speaker_frames']} chunks | ")
                sys.stdout.write(f"Detection rate: {stats['detection_percentage']:.2f}% | ")
                sys.stdout.write(f"Running: {stats['elapsed_time']:.1f}s")
                sys.stdout.flush()
                
                last_status_time = current_time
            
            time.sleep(0.1)  # Sleep to prevent high CPU usage
            
    except KeyboardInterrupt:
        # This will be caught by the signal handler
        pass
    finally:
        # Ensure cleanup happens
        detector.stop()
        detector.close()

if __name__ == "__main__":
    main()