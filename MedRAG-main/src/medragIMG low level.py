import base64
from pathlib import Path

def add_padding(base64_str):
    # Base64 strings should have a length that's a multiple of 4
    padding_needed = len(base64_str) % 4
    if padding_needed:
        base64_str += '=' * (4 - padding_needed)
    return base64_str

def convert_base64_to_image(input_file_path, output_image_path):
    try:
        input_file = Path(input_file_path)
        output_file = Path(output_image_path)

        # Check if input file exists
        if not input_file.is_file():
            print(f"Error: Input file does not exist:\n  {input_file}")
            # Optionally, list files in the directory for verification
            print(f"Listing files in the directory: {input_file.parent}")
            for file in input_file.parent.iterdir():
                print(f" - {file.name}")
            return

        # Read the Base64 text from the input file
        base64_text = input_file.read_text().strip()

        # Add padding if necessary
        base64_text = add_padding(base64_text)

        # Decode the Base64 text to binary data
        try:
            image_data = base64.b64decode(base64_text, validate=True)
        except base64.binascii.Error as decode_error:
            print(f"Error decoding Base64 data: {decode_error}")
            return

        # Detect image format by checking the header
        if image_data.startswith(b'\x89PNG\r\n\x1a\n'):
            image_format = 'png'
            print("Detected PNG format.")
        elif image_data.startswith(b'\xFF\xD8\xFF'):
            image_format = 'jpeg'
            print("Detected JPEG format.")
        elif image_data.startswith(b'GIF87a') or image_data.startswith(b'GIF89a'):
            image_format = 'gif'
            print("Detected GIF format.")
        else:
            print("Unknown image format. Saving without adding a header.")
            image_format = None

        # Optionally, adjust the output file extension based on detected format
        if image_format:
            output_image_path = Path(output_image_path).with_suffix(f'.{image_format}')

        # Write the binary data to the output image file
        output_file = Path(output_image_path)
        output_file.write_bytes(image_data)

        print(f"Image successfully saved to {output_image_path}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Define the input and output file paths
    input_file_path = r"C:\Users\drewr\Desktop\Code\code for cap\terminal output ex for pres.txt"
    output_image_path = r"C:\Users\drewr\Desktop\Code\code for cap\output_image.png"

    convert_base64_to_image(input_file_path, output_image_path)
