import argparse
import os
import sys
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
    Create a Gradio Blocks layout mirroring your Llama Vision UI,
    but with file-selector controls on the left and output on the right.
    """
    def _process(images, texts, extra_text):
        image_paths = [f.name for f in (images or [])]
        text_paths  = [f.name for f in (texts or [])]

        if extra_text:
            tmp = tempfile.NamedTemporaryFile(
                delete=False, suffix=".txt", mode="w", encoding="utf-8"
            )
            tmp.write(extra_text)
            tmp.close()
            text_paths.append(tmp.name)

        return process_selected_files(image_paths, text_paths)

    with gr.Blocks(title="Radiology File Selector") as demo:
        # ---- Header ----
        gr.HTML("<h1 style='text-align: center'>Radiology File Selector</h1>")

        # ---- Two-column row ----
        with gr.Row():
            # Left column: file inputs + extra text + button
            with gr.Column(scale=1):
                gr.Markdown("### Inputs")
                image_input = gr.File(
                    label="Radiology Images",
                    file_count="multiple",
                    file_types=[".png", ".jpg", ".jpeg", ".bmp", ".gif"]
                )
                text_input = gr.File(
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

        # ---- Wire up the button ----
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
