import cv2
import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
from PIL import Image, ImageTk
import threading
from moviepy.editor import VideoFileClip
import tempfile
import os

class VideoTextRemover:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Text Removal Tool")
        
        # Video properties
        self.video_path = None
        self.cap = None
        self.total_frames = 0
        self.current_frame = None
        self.processed_frames = []
        self.selection = None
        self.drawing = False
        self.roi_start = None
        self.roi_end = None
        self.temp_dir = tempfile.mkdtemp()
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Buttons frame
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.grid(row=0, column=0, columnspan=2, pady=5)
        
        ttk.Button(self.button_frame, text="Open Video", command=self.open_video).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.button_frame, text="Process Video", command=self.process_video).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.button_frame, text="Save Video", command=self.save_video).pack(side=tk.LEFT, padx=5)
        
        # Advanced options frame
        self.options_frame = ttk.LabelFrame(self.main_frame, text="Processing Options", padding="5")
        self.options_frame.grid(row=1, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        # Inpainting radius
        ttk.Label(self.options_frame, text="Inpainting Radius:").grid(row=0, column=0, padx=5)
        self.inpaint_radius = ttk.Scale(self.options_frame, from_=1, to=10, orient=tk.HORIZONTAL)
        self.inpaint_radius.set(3)
        self.inpaint_radius.grid(row=0, column=1, padx=5, sticky=(tk.W, tk.E))
        
        # Quality preservation
        ttk.Label(self.options_frame, text="Output Quality (Mbps):").grid(row=1, column=0, padx=5)
        self.quality_var = tk.IntVar(value=20)
        self.quality_scale = ttk.Scale(self.options_frame, from_=1, to=50, variable=self.quality_var, orient=tk.HORIZONTAL)
        self.quality_scale.grid(row=1, column=1, padx=5, sticky=(tk.W, tk.E))
        
        # Canvas for video display
        self.canvas = tk.Canvas(self.main_frame, width=800, height=600, bg='black')
        self.canvas.grid(row=2, column=0, columnspan=2, pady=5)
        self.canvas.bind("<ButtonPress-1>", self.start_selection)
        self.canvas.bind("<B1-Motion>", self.update_selection)
        self.canvas.bind("<ButtonRelease-1>", self.end_selection)
        
        # Progress bar and status
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.status_label = ttk.Label(self.main_frame, text="Ready")
        self.status_label.grid(row=4, column=0, columnspan=2, pady=5)

    def open_video(self):
        self.video_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov")]
        )
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.show_frame()
            self.status_label.config(text="Video loaded. Select text region to remove.")

    def show_frame(self):
        if self.cap is None:
            return
            
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.display_frame()

    def display_frame(self):
        if self.current_frame is None:
            return
            
        height, width = self.current_frame.shape[:2]
        canvas_ratio = 800/600
        image_ratio = width/height
        
        if image_ratio > canvas_ratio:
            new_width = 800
            new_height = int(800/image_ratio)
        else:
            new_height = 600
            new_width = int(600*image_ratio)
            
        image = Image.fromarray(self.current_frame)
        image = image.resize((new_width, new_height))
        self.photo = ImageTk.PhotoImage(image=image)
        self.canvas.create_image(400, 300, anchor=tk.CENTER, image=self.photo)

    def start_selection(self, event):
        self.drawing = True
        self.roi_start = (event.x, event.y)
        if self.selection:
            self.canvas.delete(self.selection)

    def update_selection(self, event):
        if self.drawing:
            if self.selection:
                self.canvas.delete(self.selection)
            self.selection = self.canvas.create_rectangle(
                self.roi_start[0], self.roi_start[1],
                event.x, event.y,
                outline='red', width=2
            )

    def end_selection(self, event):
        self.drawing = False
        self.roi_end = (event.x, event.y)

    def process_video(self):
        if self.cap is None or self.roi_start is None or self.roi_end is None:
            self.status_label.config(text="Please load a video and select text region first")
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        processing_thread = threading.Thread(target=self.process_frames)
        processing_thread.start()

    def process_frames(self):
        frame_count = 0
        temp_output_path = os.path.join(self.temp_dir, "temp_processed.mp4")
        
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output_path, fourcc, self.video_fps, (width, height))
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                    
                # Convert coordinates to video scale
                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()
                
                x1 = int(min(self.roi_start[0], self.roi_end[0]) * width / canvas_width)
                y1 = int(min(self.roi_start[1], self.roi_end[1]) * height / canvas_height)
                x2 = int(max(self.roi_start[0], self.roi_end[0]) * width / canvas_width)
                y2 = int(max(self.roi_start[1], self.roi_end[1]) * height / canvas_height)
                
                # Create mask for inpainting
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                mask[y1:y2, x1:x2] = 255
                
                # Inpainting
                frame = cv2.inpaint(frame, mask, int(self.inpaint_radius.get()), cv2.INPAINT_TELEA)
                out.write(frame)
                
                frame_count += 1
                progress = (frame_count / self.total_frames) * 100
                self.progress_var.set(progress)
                self.status_label.config(text=f"Processing: {progress:.1f}%")
            
            out.release()
            self.temp_video_path = temp_output_path
            self.status_label.config(text="Processing complete. You can now save the video.")
            
        except Exception as e:
            self.status_label.config(text=f"Error during processing: {str(e)}")

    def save_video(self):
        if not hasattr(self, 'temp_video_path'):
            self.status_label.config(text="No processed video to save")
            return
            
        output_path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4")]
        )
        
        if output_path:
            try:
                # Load the processed video and original audio
                video = VideoFileClip(self.temp_video_path)
                original_video = VideoFileClip(self.video_path)
                
                # Set the audio from original video
                final_video = video.set_audio(original_video.audio)
                
                # Write final video with high quality
                final_video.write_videofile(
                    output_path,
                    codec='libx264',
                    audio_codec='aac',
                    bitrate=f"{self.quality_var.get()}M",
                    preset='medium',
                    threads=4,
                    fps=self.video_fps
                )
                
                # Cleanup
                video.close()
                original_video.close()
                final_video.close()
                
                self.status_label.config(text="Video saved successfully with original audio")
                
            except Exception as e:
                self.status_label.config(text=f"Error saving video: {str(e)}")
            finally:
                # Cleanup temporary files
                if os.path.exists(self.temp_video_path):
                    os.remove(self.temp_video_path)

    def __del__(self):
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoTextRemover(root)
    root.mainloop()