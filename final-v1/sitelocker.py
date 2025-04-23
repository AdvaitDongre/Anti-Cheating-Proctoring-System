import os
import time
import psutil
import win32gui
import win32con
import ctypes
import re
import keyboard
from threading import Thread

class FullSiteLocker:
    def __init__(self, allowed_url, test_duration=10):
        """
        Initialize the FullSiteLocker with allowed URL and test duration.
        
        :param allowed_url: The only URL user is allowed to visit (e.g., 'https://example.com/test')
        :param test_duration: Duration of the test in seconds
        """
        self.allowed_url = allowed_url.lower()
        self.test_duration = test_duration
        self.running = True
        self.start_time = time.time()
        
        # Block these keys to prevent tab switching/navigation
        self.blocked_single_keys = {'tab', 'f5', 'alt', 'f4'}
        self.blocked_combinations = [
            ('ctrl', 't'),  # New tab
            ('ctrl', 'w'),  # Close tab
            ('ctrl', 'shift', 't'),  # Reopen tab
            ('ctrl', 'tab'),  # Next tab
            ('ctrl', 'shift', 'tab'),  # Previous tab
            ('alt', 'f4')  # Close window
        ]
        
        # For Windows notification
        self.user32 = ctypes.windll.user32
        
    def get_active_chrome_url(self):
        """Get the URL from the active Chrome tab."""
        try:
            window = win32gui.GetForegroundWindow()
            title = win32gui.GetWindowText(window)
            
            # Chrome window titles typically include the page title and URL
            match = re.search(r'https?://[^\s]+', title)
            if match:
                url = match.group(0).lower()
                return url
        except Exception:
            pass
        return None
    
    def is_allowed_url(self, url):
        """Check if the URL is exactly the allowed URL."""
        # if not url:
        #     return False
        # return url.startswith(self.allowed_url)
        return True
    
    def show_warning(self):
        """Show a warning to the user."""
        self.user32.MessageBoxW(0, 
                              f"You must stay on the test page at {self.allowed_url} until the test is complete!", 
                              "Test In Progress", 
                              win32con.MB_ICONWARNING)
    
    def force_return_to_allowed_url(self):
        """Attempt to force Chrome back to the allowed URL."""
        try:
            # Focus address bar (Ctrl+L)
            keyboard.press('ctrl+l')
            keyboard.release('ctrl+l')
            time.sleep(0.1)
            
            # Type the allowed URL
            keyboard.write(self.allowed_url)
            time.sleep(0.1)
            
            # Press Enter
            keyboard.press('enter')
            keyboard.release('enter')
        except Exception as e:
            print(f"Error forcing URL: {e}")
    
    def block_keys(self):
        """Block keyboard shortcuts that could change tabs or close windows."""
        # Block single keys
        for key in self.blocked_single_keys:
            keyboard.block_key(key)
        
        # Block key combinations
        for combo in self.blocked_combinations:
            keyboard.add_hotkey('+'.join(combo), lambda: None, suppress=True)
    
    def monitor(self):
        """Main monitoring loop."""
        self.block_keys()
        
        while self.running:
            # Check if test duration has elapsed
            if time.time() - self.start_time > self.test_duration:
                self.running = False
                break
                
            # Check current URL
            url = self.get_active_chrome_url()
            if not self.is_allowed_url(url):
                self.show_warning()
                self.force_return_to_allowed_url()
            
            time.sleep(0.5)  # Check frequently
    
    def start(self):
        """Start the monitoring in a separate thread."""
        self.thread = Thread(target=self.monitor)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        """Stop the monitoring."""
        self.running = False
        keyboard.unhook_all()  # Release all blocked keys
        if self.thread.is_alive():
            self.thread.join()
        print("Test completed. All restrictions lifted.")

if __name__ == "__main__":
    # Configuration
    ALLOWED_URL = "www.google.com"  # Only this exact URL will be allowed
    TEST_DURATION = 5  # seconds
    
    print(f"Starting test lockdown for {TEST_DURATION} seconds...")
    print(f"Only {ALLOWED_URL} will be accessible during the test.")
    
    locker = FullSiteLocker(ALLOWED_URL, TEST_DURATION)
    locker.start()
    
    try:
        while time.time() - locker.start_time < TEST_DURATION:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        locker.stop()