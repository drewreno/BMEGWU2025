import argparse
import tempfile

from accelerate import Accelerator
import gradio as gr

# lazy import of your processing function
from IMG_UI import process_selected_files  

# Initialize accelerator
accelerator = Accelerator()

### RUN ###
# python Final_UI.py --gradio_ui
# http://localhost:9801

def gradio_interface():
    """
    Create a Gradio Blocks layout with Ocean theme,
    file-selector controls on the left, output on the right,
    and a persistent image preview gallery.
    """
    def _process(image_files, text_files, extra_text):
        # Convert list of file objects to file paths
        image_paths = [f.name for f in (image_files or [])]
        text_paths  = [f.name for f in (text_files or [])]

        if extra_text:
            tmp = tempfile.NamedTemporaryFile(
                delete=False, suffix=".txt", mode="w", encoding="utf-8"
            )
            tmp.write(extra_text)
            tmp.close()
            text_paths.append(tmp.name)

        return process_selected_files(image_paths, text_paths)

    with gr.Blocks(title="Radiology File Selector", theme=gr.themes.Ocean()) as demo:
        # ---- Header ----
        gr.HTML("<h1 style='text-align: center; color: white;'>Radiology File Selector</h1>")

        # ---- Two-column row ----
        with gr.Row():
            # Left column: file inputs + gallery + extra text + button
            with gr.Column(scale=1):
                gr.Markdown("### Inputs", elem_id="input-header")

                # Image files uploader with preview gallery
                image_input = gr.Files(
                    label="Radiology Images",
                    file_count="multiple",
                    file_types=[".png", ".jpg", ".jpeg", ".bmp", ".gif"]
                )
                # Gallery with 2 columns to preview uploaded images
                gallery = gr.Gallery(
                    label="Preview of Uploaded Images",
                    columns=2
                )

                # Report files input
                text_input = gr.Files(
                    label="Report Files",
                    file_count="multiple",
                    file_types=[".txt", ".md", ".csv", ".log"]
                )

                extra_text = gr.Textbox(
                    label="Additional Notes",
                    placeholder="Type any extra notes here…",
                    lines=5
                )
                cont_btn = gr.Button("Continue", variant="primary")

            # Right column: processing output
            with gr.Column(scale=2):
                gr.Markdown("### Output")
                output = gr.Textbox(
                    placeholder="Results will appear here…",
                    lines=20,
                    interactive=False
                )

        # ---- Event handlers ----
        # Update gallery when image files are uploaded
        image_input.change(
            fn=lambda files: [f.name for f in (files or [])],
            inputs=image_input,
            outputs=gallery
        )

        # Process inputs when button is clicked
        cont_btn.click(
            fn=_process,
            inputs=[image_input, text_input, extra_text],
            outputs=output
        )

    return demo


def main():
    parser = argparse.ArgumentParser(
        description="Radiology File Selector (via Gradio UI)"
    )
    parser.add_argument(
        "--gradio_ui", action="store_true",
        help="Launch Gradio-based file selector UI"
    )
    args = parser.parse_args()

    if args.gradio_ui:
        demo = gradio_interface()
        demo.launch(server_name="0.0.0.0", server_port=9801)
    else:
        print("No UI mode selected. Use --gradio_ui to launch the Gradio interface.")


if __name__ == "__main__":
    main()
