import gradio as gr

from compress import ICompressor

from compress import ZLibCompressor

compressor = ZLibCompressor()


# Function to handle the compress button logic
def handle_compress(text):
    compression_ratio, compressed_bytes = compressor.compress(text)
    file_name = "compressed_data.bin"
    with open(file_name, "wb") as f:
        f.write(compressed_bytes)
    return compression_ratio, file_name


# Function to handle file upload for decompression
def handle_decompress(file):
    with open(file.name, "rb") as f:
        compressed_bytes = f.read()
    return compressor.decompress(compressed_bytes)


if __name__ == '__main__':
    with gr.Blocks() as demo:
        # Textbox for input
        with gr.Row():
            input_text = gr.Textbox(label="Enter Text to Compress", lines=5)

        # Button for compression
        with gr.Row():
            compress_button = gr.Button("Compress")

        # Output fields for compression
        with gr.Row():
            compression_ratio = gr.Number(label="Compression Ratio")
            download_file = gr.File(label="Download Compressed File")

        compress_button.click(
            fn=handle_compress,
            inputs=[input_text],
            outputs=[compression_ratio, download_file]
        )

        # File uploader for decompression
        with gr.Row():
            compressed_file = gr.File(label="Upload Compressed File")
            decompress_button = gr.Button("Decompress")

        # Output field for decompressed text
        with gr.Row():
            decompressed_text = gr.Textbox(label="Decompressed Text", lines=5)

        decompress_button.click(
            fn=handle_decompress,
            inputs=[compressed_file],
            outputs=[decompressed_text]
        )

    demo.launch()
