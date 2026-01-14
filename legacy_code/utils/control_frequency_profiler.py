"""
Control frequency profiler for real robot control.

Measures actual command send frequency and identifies bottlenecks in the control pipeline.
Can be used as a callback or decorator to profile the control loop.

Usage:
    # Option 1: As a callback in the main control loop
    profiler = ControlFrequencyProfiler()
    profiler.start()
    
    # In control loop:
    profiler.before_send()
    robot_driver.send_motor_positions(...)
    profiler.after_send()
    
    # Periodically print stats:
    profiler.print_stats()
    
    # Option 2: As a context manager
    with ControlFrequencyProfiler() as profiler:
        # control loop here
        profiler.before_send()
        robot_driver.send_motor_positions(...)
        profiler.after_send()
"""
import time
import collections
from typing import Optional


class ControlFrequencyProfiler:
    """
    Profiles control loop frequency and command send timing.
    
    Tracks:
    - Actual command send frequency (Hz)
    - Time to send commands (ms)
    - Loop iteration time (ms)
    - Bottlenecks in the pipeline
    """
    
    def __init__(self, stats_interval: int = 100):
        """
        Initialize profiler.
        
        Args:
            stats_interval: Print stats every N command sends
        """
        self.stats_interval = stats_interval
        
        # Timing data
        self.send_times = collections.deque(maxlen=1000)  # Time to send commands
        self.interval_times = collections.deque(maxlen=1000)  # Time between sends
        self.loop_times = collections.deque(maxlen=1000)  # Full loop iteration time
        
        # State
        self.last_send_time: Optional[float] = None
        self.send_count = 0
        self.loop_start_time: Optional[float] = None
        self.send_start_time: Optional[float] = None
        
        # Pipeline timing (optional, for detailed breakdown)
        self.pipeline_times = {
            'spacemouse_update': [],
            'ik_solve': [],
            'joint_to_motor': [],
            'send_commands': [],
        }
        
        self.running = False
    
    def start(self):
        """Start profiling."""
        self.running = True
        self.last_send_time = time.perf_counter()
        print("Control frequency profiler started")
        print(f"Will print stats every {self.stats_interval} command sends\n")
    
    def stop(self):
        """Stop profiling and print final stats."""
        self.running = False
        self.print_stats()
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
    
    def before_loop(self):
        """Call at the start of each control loop iteration."""
        if not self.running:
            return
        self.loop_start_time = time.perf_counter()
    
    def after_loop(self):
        """Call at the end of each control loop iteration."""
        if not self.running or self.loop_start_time is None:
            return
        loop_time = time.perf_counter() - self.loop_start_time
        self.loop_times.append(loop_time)
    
    def before_send(self):
        """Call immediately before sending motor commands."""
        if not self.running:
            return
        self.send_start_time = time.perf_counter()
    
    def after_send(self):
        """Call immediately after sending motor commands."""
        if not self.running or self.send_start_time is None:
            return
        
        send_end_time = time.perf_counter()
        send_duration = send_end_time - self.send_start_time
        self.send_times.append(send_duration)
        
        # Calculate interval since last send
        if self.last_send_time is not None:
            interval = send_end_time - self.last_send_time
            self.interval_times.append(interval)
        
        self.last_send_time = send_end_time
        self.send_count += 1
        
        # Print stats periodically
        if self.send_count % self.stats_interval == 0:
            self.print_stats()
    
    def record_pipeline_step(self, step_name: str, duration: float):
        """
        Record timing for a specific pipeline step.
        
        Args:
            step_name: Name of the step (e.g., 'ik_solve', 'joint_to_motor')
            duration: Duration in seconds
        """
        if not self.running:
            return
        if step_name in self.pipeline_times:
            self.pipeline_times[step_name].append(duration)
            # Keep only recent data
            if len(self.pipeline_times[step_name]) > 1000:
                self.pipeline_times[step_name] = self.pipeline_times[step_name][-1000:]
    
    def print_stats(self):
        """Print current statistics."""
        if not self.running:
            return
        
        print("\n" + "="*70)
        print("CONTROL FREQUENCY PROFILE")
        print("="*70)
        
        # Command send frequency
        if len(self.interval_times) > 0:
            avg_interval = sum(self.interval_times) / len(self.interval_times)
            actual_freq = 1.0 / avg_interval if avg_interval > 0 else 0
            min_interval = min(self.interval_times)
            max_interval = max(self.interval_times)
            min_freq = 1.0 / max_interval if max_interval > 0 else 0
            max_freq = 1.0 / min_interval if min_interval > 0 else 0
            
            print(f"\nCommand Send Frequency:")
            print(f"  Average: {actual_freq:.2f} Hz (target: 50 Hz)")
            print(f"  Range:   {min_freq:.2f} - {max_freq:.2f} Hz")
            print(f"  Interval: {avg_interval*1000:.2f} ms avg ({min_interval*1000:.2f} - {max_interval*1000:.2f} ms)")
            
            if actual_freq < 45:
                print(f"  ⚠️  WARNING: Frequency is below 45 Hz! Possible bottleneck detected.")
            elif actual_freq > 55:
                print(f"  ⚠️  WARNING: Frequency is above 55 Hz! Loop may be running too fast.")
        
        # Send command duration
        if len(self.send_times) > 0:
            avg_send = sum(self.send_times) / len(self.send_times) * 1000
            max_send = max(self.send_times) * 1000
            min_send = min(self.send_times) * 1000
            
            print(f"\nCommand Send Duration:")
            print(f"  Average: {avg_send:.2f} ms")
            print(f"  Range:   {min_send:.2f} - {max_send:.2f} ms")
            
            if avg_send > 15:
                print(f"  ⚠️  WARNING: Send duration is high! May be bottleneck.")
        
        # Loop iteration time
        if len(self.loop_times) > 0:
            avg_loop = sum(self.loop_times) / len(self.loop_times) * 1000
            max_loop = max(self.loop_times) * 1000
            min_loop = min(self.loop_times) * 1000
            
            print(f"\nLoop Iteration Time:")
            print(f"  Average: {avg_loop:.2f} ms")
            print(f"  Range:   {min_loop:.2f} - {max_loop:.2f} ms")
            print(f"  Target:  {1000/50:.2f} ms (for 50 Hz)")
            
            if avg_loop > 25:
                print(f"  ⚠️  WARNING: Loop time exceeds target! May miss 50 Hz.")
        
        # Pipeline breakdown
        if any(len(times) > 0 for times in self.pipeline_times.values()):
            print(f"\nPipeline Breakdown (average times):")
            for step_name, times in self.pipeline_times.items():
                if len(times) > 0:
                    avg_time = sum(times) / len(times) * 1000
                    max_time = max(times) * 1000
                    print(f"  {step_name:20s}: {avg_time:6.2f} ms (max: {max_time:6.2f} ms)")
        
        print(f"\nTotal commands sent: {self.send_count}")
        print("="*70 + "\n")
    
    def get_stats_dict(self):
        """
        Get statistics as a dictionary.
        
        Returns:
            dict: Statistics dictionary
        """
        stats = {
            'send_count': self.send_count,
            'send_times': list(self.send_times),
            'interval_times': list(self.interval_times),
            'loop_times': list(self.loop_times),
        }
        
        if len(self.interval_times) > 0:
            avg_interval = sum(self.interval_times) / len(self.interval_times)
            stats['actual_frequency'] = 1.0 / avg_interval if avg_interval > 0 else 0
        
        if len(self.send_times) > 0:
            stats['avg_send_duration_ms'] = sum(self.send_times) / len(self.send_times) * 1000
        
        if len(self.loop_times) > 0:
            stats['avg_loop_duration_ms'] = sum(self.loop_times) / len(self.loop_times) * 1000
        
        return stats
