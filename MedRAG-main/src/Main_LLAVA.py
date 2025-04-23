import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import logging
import threading
import json
import re
import os
import tempfile

w = 600
h = 400  # Increased height to accommodate text input
gap = 5

class FileSelectorUI:
    def __init__(self, root):
        self.root = root
        self.root.title("File Selector")
        self.root.geometry(f"{w}x{h}")
        self.root.configure(bg='white')

        # Configure rows and columns for dynamic resizing
        self.root.rowconfigure(0, weight=0)  # Text input row
        self.root.rowconfigure(1, weight=1)  # Frames row
        self.root.rowconfigure(2, weight=0)  # Buttons row
        self.root.rowconfigure(3, weight=0)
        self.root.rowconfigure(4, weight=0)
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)

        # Initialize lists to store selected files
        self.selected_images = []
        self.selected_texts = []

        # Create text input at the top
        self.create_text_input()
        # Create frames for Images and Text Files
        self.create_sections()
        # Create buttons
        self.create_buttons()

        # Bind the window close event to the on_closing method
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_text_input(self):
        input_frame = ttk.LabelFrame(self.root, text="Additional Text Input", padding=gap)
        input_frame.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=gap, pady=gap)
        self.text_input = tk.Text(input_frame, height=5)
        self.text_input.pack(fill=tk.BOTH, expand=True)

    def create_sections(self):
        # Frame for Images
        images_frame = ttk.LabelFrame(self.root, text="Selected Radiology Images", padding=gap)
        images_frame.grid(row=1, column=0, sticky="nsew", padx=gap, pady=gap)

        # Frame for Text Files
        texts_frame = ttk.LabelFrame(self.root, text="Selected Report Files", padding=gap)
        texts_frame.grid(row=1, column=1, sticky="nsew", padx=gap, pady=gap)

        # Ensure uniform column sizing
        self.root.columnconfigure(0, weight=1, uniform="group1")
        self.root.columnconfigure(1, weight=1, uniform="group1")

        # Listbox to display selected images
        self.images_listbox = tk.Listbox(images_frame, selectmode=tk.MULTIPLE)
        self.images_listbox.pack(fill=tk.BOTH, expand=True)

        # Listbox to display selected text files
        self.texts_listbox = tk.Listbox(texts_frame, selectmode=tk.MULTIPLE)
        self.texts_listbox.pack(fill=tk.BOTH, expand=True)

    def create_buttons(self):
        # Button to select images
        select_images_btn = ttk.Button(self.root, text="Select Images", command=self.select_images)
        select_images_btn.grid(row=2, column=0, padx=gap, pady=gap, sticky="nsew")

        # Button to select text files
        select_texts_btn = ttk.Button(self.root, text="Select Text Files", command=self.select_text_files)
        select_texts_btn.grid(row=2, column=1, padx=gap, pady=gap, sticky="nsew")

        # Continue button (to process selected files)
        continue_btn = ttk.Button(self.root, text="Continue", command=self.continue_action)
        continue_btn.grid(row=3, column=0, columnspan=2, padx=gap, pady=gap, sticky="nsew")

        # Exit button (to close the program)
        exit_btn = ttk.Button(self.root, text="Exit", command=self.exit_action)
        exit_btn.grid(row=4, column=0, columnspan=2, padx=gap, pady=gap, sticky="nsew")

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
        # Ensure at least one file or additional text is provided
        additional_text = self.text_input.get("1.0", tk.END).strip()
        if not (self.selected_images or self.selected_texts or additional_text):
            messagebox.showwarning("No Selection", "Please select at least one image, text file, or enter additional text.")
            return

        # If there's additional text, write it to a temporary file and include in texts
        if additional_text:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.txt', mode='w', encoding='utf-8')
            tmp.write(additional_text)
            tmp.close()
            self.selected_texts.append(tmp.name)
            self.texts_listbox.insert(tk.END, os.path.basename(tmp.name))

        # Confirm the action
        confirm = messagebox.askyesno("Confirm", "Do you want to continue with the selected files and text?" )
        if not confirm:
            return

        # Disable buttons during processing
        self.disable_buttons()

        # Show a loading message in a new window
        loading = tk.Toplevel(self.root)
        loading.title("Processing")
        loading.geometry("300x100")
        loading_label = tk.Label(loading, text="Processing selected files...Please wait.", padx=20, pady=20)
        loading_label.pack()

        # Process files in a separate thread so the UI remains responsive
        thread = threading.Thread(target=self.process_files, args=(loading,), daemon=True)
        thread.start()

    def process_files(self, loading_window):
        try:
            # Lazy import of process_selected_files to speed up startup
            from LLAVA import process_selected_files
            output = process_selected_files(self.selected_images, self.selected_texts)
            # Schedule the success handling in the main thread
            self.root.after(0, lambda: self.on_processing_complete(loading_window, output))
        except Exception as e:
            logging.error(f"An error occurred during processing: {e}")
            # Schedule the error handling in the main thread
            self.root.after(0, lambda e=e: self.on_processing_error(loading_window, str(e)))

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

        # Create a scrollable result window
        result_window = tk.Toplevel(self.root)
        result_window.title("Processing Complete")
        result_window.geometry("600x400")

        container = ttk.Frame(result_window)
        container.pack(fill='both', expand=True, padx=5, pady=5)

        text_widget = tk.Text(container, wrap='word')
        text_widget.insert('1.0', cleaned_json)
        text_widget.config(state='disabled')

        vsb = ttk.Scrollbar(container, orient='vertical', command=text_widget.yview)
        text_widget['yscrollcommand'] = vsb.set

        vsb.pack(side='right', fill='y')
        text_widget.pack(side='left', fill='both', expand=True)

        close_btn = ttk.Button(result_window, text="Close", command=result_window.destroy)
        close_btn.pack(pady=(0,5))

        # Clear file selections and text input, then re-enable buttons
        self.selected_images = []
        self.selected_texts = []
        self.update_listbox(self.images_listbox, [])
        self.update_listbox(self.texts_listbox, [])
        self.text_input.delete('1.0', tk.END)
        self.enable_buttons()

    def on_processing_error(self, loading_window, error_message):
        loading_window.destroy()
        messagebox.showerror("Error", f"An error occurred: {error_message}")
        self.enable_buttons()

    def disable_buttons(self):
        for btn in self.buttons:
            btn.config(state='disabled')

    def enable_buttons(self):
        for btn in self.buttons:
            btn.config(state='normal')

    def exit_action(self):
        if messagebox.askokcancel("Exit", "Do you really want to exit?" ):
            self.root.destroy()

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?" ):
            self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FileSelectorUI(root)
    root.mainloop()
