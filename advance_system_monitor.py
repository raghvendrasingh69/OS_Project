import tkinter as tk
from tkinter import ttk
import psutil
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time

class SystemOptimizerDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("AI System Call Optimizer")
        self.root.geometry("1200x800")
        
        # Data storage
        self.data = pd.DataFrame(columns=['timestamp', 'cpu', 'memory', 'disk_read', 'disk_write'])
        self.cluster_labels = []
        
        # Create dashboard frames
        self.create_control_panel()
        self.create_metrics_display()
        self.create_visualization_frame()
        
        # Start monitoring thread
        self.monitor_active = True
        self.monitor_thread = threading.Thread(target=self.monitor_system)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def create_control_panel(self):
        control_frame = ttk.LabelFrame(self.root, text="Control Panel", padding=10)
        control_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        ttk.Button(control_frame, text="Start Optimization", command=self.start_optimization).pack(pady=5)
        ttk.Button(control_frame, text="Pause Monitoring", command=self.toggle_monitoring).pack(pady=5)
        ttk.Button(control_frame, text="View Recommendations", command=self.show_recommendations).pack(pady=5)
        
        self.cluster_var = tk.IntVar(value=1)
        ttk.Checkbutton(control_frame, text="Show Clusters", variable=self.cluster_var).pack(pady=5)
        
    def create_metrics_display(self):
        metrics_frame = ttk.LabelFrame(self.root, text="System Metrics", padding=10)
        metrics_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        self.cpu_label = ttk.Label(metrics_frame, text="CPU: 0%")
        self.mem_label = ttk.Label(metrics_frame, text="Memory: 0%")
        self.disk_label = ttk.Label(metrics_frame, text="Disk I/O: 0MB/s")
        self.net_label = ttk.Label(metrics_frame, text="Network: 0KB/s")
        
        for label in [self.cpu_label, self.mem_label, self.disk_label, self.net_label]:
            label.pack(pady=5)
        
        self.status_label = ttk.Label(metrics_frame, text="Status: Monitoring")
        self.status_label.pack(pady=10)
        
    def create_visualization_frame(self):
        vis_frame = ttk.LabelFrame(self.root, text="Performance Analysis", padding=10)
        vis_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        
        self.fig, self.ax = plt.subplots(2, 2, figsize=(10, 8))
        plt.subplots_adjust(hspace=0.5)
        
        # Create plots
        plots = [
            (self.ax[0,0], 'r-', 'CPU Usage (%)'),
            (self.ax[0,1], 'b-', 'Memory Usage (%)'),
            (self.ax[1,0], 'g-', 'Disk I/O (MB/s)')
        ]
        
        self.plot_lines = []
        for ax, style, title in plots:
            line, = ax.plot([], [], style)
            ax.set_title(title)
            self.plot_lines.append(line)
        
        self.cluster_scatter = self.ax[1,1].scatter([], [], c=[], cmap='viridis')
        self.ax[1,1].set_title('Behavioral Clusters')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=vis_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def monitor_system(self):
        while self.monitor_active:
            try:
                cpu = psutil.cpu_percent()
                memory = psutil.virtual_memory().percent
                disk = psutil.disk_io_counters()
                net = psutil.net_io_counters()
                
                # Update UI
                self.cpu_label.config(text=f"CPU: {cpu:.1f}%")
                self.mem_label.config(text=f"Memory: {memory:.1f}%")
                self.disk_label.config(text=f"Disk I/O: {disk.read_bytes/1e6:.1f}MB/s")
                self.net_label.config(text=f"Network: {net.bytes_recv/1e3:.1f}KB/s")
                
                # Store data
                new_row = {
                    'timestamp': time.time(),
                    'cpu': cpu,
                    'memory': memory,
                    'disk_read': disk.read_bytes,
                    'disk_write': disk.write_bytes
                }
                self.data = pd.concat([self.data, pd.DataFrame([new_row])], ignore_index=True)
                
                if len(self.data) > 60:
                    self.data = self.data.iloc[-60:]
                
                self.update_plots()
                time.sleep(1)
            except Exception as e:
                print(f"Monitoring error: {e}")
                break
    
    def update_plots(self):
        x = range(len(self.data))
        
        # Update time series plots
        metrics = ['cpu', 'memory', 'disk_read']
        for line, metric in zip(self.plot_lines, metrics):
            y = self.data[metric] if metric != 'disk_read' else self.data[metric]/1e6
            line.set_data(x, y)
            line.axes.relim()
            line.axes.autoscale_view()
        
        # Update clusters
        if self.cluster_var.get() and len(self.data) > 10:
            self.perform_clustering()
            self.cluster_scatter.set_offsets(self.data[['cpu', 'memory']].values)
            self.cluster_scatter.set_array(self.cluster_labels)
            self.ax[1,1].relim()
            self.ax[1,1].autoscale_view()
        
        self.canvas.draw()
    
    def perform_clustering(self):
        recent_data = self.data.iloc[-30:][['cpu', 'memory', 'disk_read']]
        normalized = (recent_data - recent_data.mean()) / recent_data.std()
        kmeans = KMeans(n_clusters=3)
        self.cluster_labels = kmeans.fit_predict(normalized)
    
    def start_optimization(self):
        self.status_label.config(text="Status: Optimizing")
        print("Optimization logic would run here")
    
    def toggle_monitoring(self):
        self.monitor_active = not self.monitor_active
        status = "Paused" if not self.monitor_active else "Monitoring"
        self.status_label.config(text=f"Status: {status}")
    
    def show_recommendations(self):
        popup = tk.Toplevel()
        popup.title("Recommendations")
        
        if len(self.data) < 5:
            ttk.Label(popup, text="Insufficient data").pack(pady=20)
        else:
            avg_cpu = self.data['cpu'].mean()
            avg_mem = self.data['memory'].mean()
            
            recommendations = []
            if avg_cpu > 70:
                recommendations.append("High CPU: Close background apps")
            if avg_mem > 75:
                recommendations.append("High RAM: Check for memory leaks")
            
            if not recommendations:
                recommendations.append("System is well optimized")
            
            for i, rec in enumerate(recommendations, 1):
                ttk.Label(popup, text=f"{i}. {rec}").pack(anchor='w', padx=20, pady=5)
        
        ttk.Button(popup, text="Close", command=popup.destroy).pack(pady=10)

if __name__ == "__main__":
    root = tk.Tk()
    try:
        app = SystemOptimizerDashboard(root)
        root.mainloop()
    except ImportError as e:
        print(f"Required package missing: {e}\n"
              "Install with: pip install psutil pandas scikit-learn matplotlib")