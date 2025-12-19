"""
Simple GUI for robot control commands.

Provides a lightweight tkinter-based GUI with buttons for robot control:
- Home: Move robot to home position
- Reset EE: Reboot & open gripper (no arm motion)
- Start Recording: Begin/reset trajectory recording
- Stop & Save: Stop recording and save trajectory
- Stop & Discard: Stop recording and discard current trajectory
"""
import time
import threading


class SimpleControlGUI:
    """Simple GUI with buttons for robot control."""
    
    def __init__(self):
        self.command_queue = []
        self.lock = threading.Lock()
        self.root = None
        self.thread = None
        self.running = False
        self.status_label = None
        
    def _gui_thread(self):
        """Run GUI in a separate thread."""
        try:
            import tkinter as tk
            from tkinter import ttk
            
            self.root = tk.Tk()
            self.root.title("WX200 Robot Control")
            # Slightly taller to comfortably fit 5 buttons + status label
            self.root.geometry("340x240")
            self.root.resizable(False, False)
            
            # Handle window close - mark as not running but don't destroy yet
            def on_closing():
                self.running = False
                self.root.quit()
            
            self.root.protocol("WM_DELETE_WINDOW", on_closing)
            
            # Handle thread-safe quit signal
            self.root.bind("<<Quit>>", lambda e: self.root.quit())
            
            # Configure style
            style = ttk.Style()
            style.theme_use('clam')
            
            # Create buttons
            frame = ttk.Frame(self.root, padding="10")
            frame.pack(fill=tk.BOTH, expand=True)
            
            # Home button
            home_btn = ttk.Button(
                frame, 
                text="Home (h)", 
                command=lambda: self._queue_command('h'),
                width=20
            )
            home_btn.pack(pady=5)
            
            # Reset EE (gripper) button
            reset_ee_btn = ttk.Button(
                frame,
                text="Reset EE (g)",
                command=lambda: self._queue_command('g'),
                width=20
            )
            reset_ee_btn.pack(pady=5)
            
            # Start Recording button
            record_btn = ttk.Button(
                frame, 
                text="Start Recording (r)", 
                command=lambda: self._queue_command('r'),
                width=20
            )
            record_btn.pack(pady=5)
            
            # Stop & Save button
            save_btn = ttk.Button(
                frame, 
                text="Stop & Save (s)", 
                command=lambda: self._queue_command('s'),
                width=20
            )
            save_btn.pack(pady=5)
            
            # Stop & Discard button
            discard_btn = ttk.Button(
                frame,
                text="Stop & Discard (d)",
                command=lambda: self._queue_command('d'),
                width=20
            )
            discard_btn.pack(pady=5)
            
            # Status label
            self.status_label = ttk.Label(frame, text="Ready", foreground="green")
            self.status_label.pack(pady=5)
            
            self.running = True
            self.root.mainloop()
            
            # Cleanup: destroy root in the GUI thread
            if self.root:
                try:
                    self.root.destroy()
                except:
                    pass
                self.root = None
            
        except Exception as e:
            print(f"⚠️  GUI error: {e}")
            self.running = False
            if self.root:
                try:
                    self.root.destroy()
                except:
                    pass
                self.root = None
    
    def _queue_command(self, command):
        """Queue a command and update status."""
        with self.lock:
            self.command_queue.append(command)
        if self.status_label:
            self.status_label.config(text=f"Command: {command}", foreground="blue")
            self.root.after(1000, lambda: self.status_label.config(text="Ready", foreground="green"))
    
    def start(self):
        """Start the GUI in a separate thread."""
        try:
            # Use non-daemon thread so it can be properly cleaned up
            self.thread = threading.Thread(target=self._gui_thread, daemon=False)
            self.thread.start()
            # Give GUI time to initialize
            time.sleep(0.5)
        except Exception as e:
            print(f"⚠️  Failed to start GUI: {e}")
            self.running = False
    
    def stop(self):
        """Stop the GUI and wait for thread to finish."""
        self.running = False
        if self.root:
            try:
                # Schedule quit in the GUI thread safely using virtual event
                self.root.event_generate("<<Quit>>")
            except:
                pass
        
        # Wait for GUI thread to finish (with timeout)
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
    
    def get_command(self):
        """Get a command from the queue if available, returns None otherwise."""
        with self.lock:
            if self.command_queue:
                return self.command_queue.pop(0)
        return None
    
    def is_available(self):
        """Check if GUI is available and running."""
        return self.running and self.root is not None
