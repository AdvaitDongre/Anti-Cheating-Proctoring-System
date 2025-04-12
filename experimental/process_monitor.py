import psutil
import time
from datetime import datetime
import os
import hashlib

class AdvancedProcessMonitor:
    def __init__(self):
        self.scan_interval = 5  # seconds
        self.log_file = "suspicious_processes.log"
        
        # **Whitelist Approaches**
        self.system_whitelist = self._get_core_system_processes()
        self.trusted_publishers = ["Microsoft", "Google", "Mozilla", "Intel", "NVIDIA"]
        
        # **Malware Detection Heuristics**
        self.suspicious_keywords = ["cheat", "hack", "trainer", "inject", "spoofer"]
        self.temp_file_indicators = [".tmp", "temp", "~"]
        
        # **Process Behavior Checks**
        self.high_cpu_threshold = 50  # %
        self.high_memory_threshold = 500  # MB

    def _get_core_system_processes(self):
        """Returns critical OS processes that should never be flagged."""
        return {
            "svchost.exe", "explorer.exe", "wininit.exe", "smss.exe",
            "csrss.exe", "lsass.exe", "services.exe", "taskhostw.exe",
            "dwm.exe", "ctfmon.exe", "spoolsv.exe", "SearchIndexer.exe",
            "RuntimeBroker.exe", "conhost.exe", "audiodg.exe", "dllhost.exe",
            "sihost.exe", "fontdrvhost.exe", "WmiPrvSE.exe", "MemCompression",
            "SecurityHealthService.exe", "ShellExperienceHost.exe", "Registry"
        }

    def _is_trusted_publisher(self, process_path):
        """Checks if the process is signed by a trusted publisher."""
        if not process_path or not os.path.exists(process_path):
            return False
            
        try:
            import win32api  # Only works on Windows (pywin32)
            info = win32api.GetFileVersionInfo(process_path, "\\")
            company_name = info.get("CompanyName", "").lower()
            return any(pub.lower() in company_name for pub in self.trusted_publishers)
        except:
            return False

    def _is_suspicious_name(self, process_name):
        """Checks for known malware naming patterns."""
        lower_name = process_name.lower()
        return any(keyword in lower_name for keyword in self.suspicious_keywords)

    def _is_running_from_temp(self, process_path):
        """Detects if a process is running from a suspicious location."""
        if not process_path:
            return False
        temp_dirs = ["temp", "tmp", "appdata\\local\\temp", "downloads"]
        return any(dir in process_path.lower() for dir in temp_dirs)

    def _is_high_resource_usage(self, process):
        """Flags processes consuming excessive CPU/Memory."""
        try:
            cpu_percent = process.cpu_percent(interval=0.1)
            mem_usage = process.memory_info().rss / (1024 * 1024)  # MB
            return (cpu_percent > self.high_cpu_threshold or 
                    mem_usage > self.high_memory_threshold)
        except:
            return False

    def scan_processes(self):
        suspicious = []
        
        for proc in psutil.process_iter(['pid', 'name', 'exe', 'username']):
            try:
                proc_info = proc.info
                proc_name = proc_info['name']
                proc_path = proc_info.get('exe', '')
                
                # **Skip Whitelisted Processes**
                if proc_name in self.system_whitelist:
                    continue
                    
                # **Check Trusted Publishers (Windows Only)**
                if self._is_trusted_publisher(proc_path):
                    continue
                    
                # **Detection Flags**
                flags = []
                if self._is_suspicious_name(proc_name):
                    flags.append("Suspicious name")
                if self._is_running_from_temp(proc_path):
                    flags.append("Running from temp location")
                if self._is_high_resource_usage(proc):
                    flags.append("High resource usage")
                
                if flags:
                    suspicious.append({
                        'name': proc_name,
                        'pid': proc_info['pid'],
                        'path': proc_path,
                        'flags': " | ".join(flags)
                    })
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        return suspicious

    def log_findings(self, suspicious_processes):
        if not suspicious_processes:
            return
            
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"\n[{timestamp}] SUSPICIOUS PROCESSES DETECTED:\n"
        
        for proc in suspicious_processes:
            log_entry += (
                f" - {proc['name']} (PID: {proc['pid']})\n"
                f"   Path: {proc['path']}\n"
                f"   Reason: {proc['flags']}\n"
            )
        
        print(log_entry)
        with open(self.log_file, "a") as f:
            f.write(log_entry)

    def monitor(self):
        try:
            while True:
                suspicious = self.scan_processes()
                self.log_findings(suspicious)
                time.sleep(self.scan_interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped. Log saved to:", os.path.abspath(self.log_file))

if __name__ == "__main__":
    monitor = AdvancedProcessMonitor()
    monitor.monitor()