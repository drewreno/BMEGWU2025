import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import logging
import threading
import json
import re
import os

w = 600
h = 300
gap = 5


class FileSelectorUI:
    def __init__(self, root):
        self.root = root
        self.root.title("File Selector")
        self.root.geometry(f"{w}x{h}")
        self.root.configure(bg='white')

        # Configure rows and columns for dynamic resizing
        self.root.rowconfigure(0, weight=1)  # Frames row
        self.root.rowconfigure(1, weight=0)  # Buttons row
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)

        # Initialize lists to store selected files
        self.selected_images = []
        self.selected_texts = []

        # Create frames for Images and Text Files
        self.create_sections()

        # Create buttons
        self.create_buttons()

        # Bind the window close event to the on_closing method
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_sections(self):
        # Frame for Images
        images_frame = ttk.LabelFrame(self.root, text="Selected Radiology Images", padding=gap)
        images_frame.grid(row=0, column=0, sticky="nsew")

        # Frame for Text Files
        texts_frame = ttk.LabelFrame(self.root, text="Selected Report Files", padding=gap)
        texts_frame.grid(row=0, column=1, sticky="nsew")

        # Ensure uniform column sizing
        self.root.columnconfigure(0, weight=1, uniform="group1")
        self.root.columnconfigure(1, weight=1, uniform="group1")
        self.root.rowconfigure(0, weight=1)

        # Listbox to display selected images
        self.images_listbox = tk.Listbox(images_frame, selectmode=tk.MULTIPLE)
        self.images_listbox.pack(fill=tk.BOTH, expand=True)

        # Listbox to display selected text files
        self.texts_listbox = tk.Listbox(texts_frame, selectmode=tk.MULTIPLE)
        self.texts_listbox.pack(fill=tk.BOTH, expand=True)

    def create_buttons(self):
        # Button to select images
        select_images_btn = ttk.Button(self.root, text="Select Images", command=self.select_images)
        select_images_btn.grid(row=1, column=0, padx=gap, pady=gap, sticky="nsew")

        # Button to select text files
        select_texts_btn = ttk.Button(self.root, text="Select Text Files", command=self.select_text_files)
        select_texts_btn.grid(row=1, column=1, padx=gap, pady=gap, sticky="nsew")

        # Continue button (to process selected files)
        continue_btn = ttk.Button(self.root, text="Continue", command=self.continue_action)
        continue_btn.grid(row=2, column=0, columnspan=2, padx=gap, pady=gap, sticky="nsew")

        # Exit button (to close the program)
        exit_btn = ttk.Button(self.root, text="Exit", command=self.exit_action)
        exit_btn.grid(row=3, column=0, columnspan=2, padx=gap, pady=gap, sticky="nsew")

        # Store buttons (we disable these while processing)
        self.buttons = [select_images_btn, select_texts_btn, continue_btn]

    def select_images(self):
        file_paths = filedialog.askopenfilenames(
            title="Select Image Files",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.gif *.bmp")]
        )
        if file_paths:
            self.selected_images = list(file_paths)
            file_names = [os.path.basename(path) for path in self.selected_images]
            self.update_listbox(self.images_listbox, file_names)

    def select_text_files(self):
        file_paths = filedialog.askopenfilenames(
            title="Select Text Files",
            filetypes=[("Text Files", "*.txt *.md *.csv *.log")]
        )
        if file_paths:
            self.selected_texts = list(file_paths)
            file_names = [os.path.basename(path) for path in self.selected_texts]
            self.update_listbox(self.texts_listbox, file_names)

    def update_listbox(self, listbox, items):
        listbox.delete(0, tk.END)
        for item in items:
            listbox.insert(tk.END, item)

    def continue_action(self):
        # Make sure some files have been selected
        if not self.selected_images and not self.selected_texts:
            messagebox.showwarning("No Selection", "Please select at least one image or text file.")
            return

        # Confirm the action
        confirm = messagebox.askyesno("Confirm", "Do you want to continue with the selected files?")
        if not confirm:
            return

        # Disable buttons during processing
        self.disable_buttons()

        # Show a loading message in a new window
        loading = tk.Toplevel(self.root)
        loading.title("Processing")
        loading.geometry("300x100")
        loading_label = tk.Label(loading, text="Processing selected files...\nPlease wait.", padx=20, pady=20)
        loading_label.pack()

        # Process files in a separate thread so the UI remains responsive
        thread = threading.Thread(target=self.process_files, args=(loading,), daemon=True)
        thread.start()

    def process_files(self, loading_window):
        try:
            # Lazy import of process_selected_files to speed up startup
            from IMG_UI import process_selected_files
            output = process_selected_files(self.selected_images, self.selected_texts)
            # Schedule the success handling in the main thread
            self.root.after(0, lambda: self.on_processing_complete(loading_window, output))
        except Exception as e:
            logging.error(f"An error occurred during processing: {e}")
            # Schedule the error handling in the main thread
            self.root.after(0, lambda: self.on_processing_error(loading_window, str(e)))

    def on_processing_complete(self, loading_window, answer):
        loading_window.destroy()

        # Check for JSON content in the answer and format it
        pattern = r"```json\s*(\{.*?\})\s*```"
        match = re.search(pattern, answer, re.DOTALL)
        if match:
            json_content = match.group(1)
            try:
                parsed_json = json.loads(json_content)
                cleaned_json = json.dumps(parsed_json, indent=4)
                with open('output.txt', 'w') as f:
                    f.write("Response:\n")
                    f.write(cleaned_json)
            except json.JSONDecodeError:
                cleaned_json = answer
        else:
            cleaned_json = answer
            with open('output.txt', 'w') as f:
                f.write(f"Response: {answer}\n")

        # Display the output to the user
        messagebox.showinfo("Processing Complete", f"Response:\n{cleaned_json}")

        # Clear file selections so new ones can be made for the next run
        self.selected_images = []
        self.selected_texts = []
        self.update_listbox(self.images_listbox, [])
        self.update_listbox(self.texts_listbox, [])

        # Re-enable buttons for further use
        self.enable_buttons()

    def on_processing_error(self, loading_window, error_message):
        loading_window.destroy()
        messagebox.showerror("Error", f"An error occurred: {error_message}")
        # Re-enable buttons so the user can try again
        self.enable_buttons()

    def disable_buttons(self):
        for btn in self.buttons:
            btn.config(state='disabled')

    def enable_buttons(self):
        for btn in self.buttons:
            btn.config(state='normal')

    def exit_action(self):
        if messagebox.askokcancel("Exit", "Do you really want to exit?"):
            self.root.destroy()

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = FileSelectorUI(root)
    root.mainloop()
