# Music Transformer with 🤗 Transformers

This project implements the Music Transformer, a powerful attention-based neural network for generating long pieces of music with improved long-term coherence. The Music Transformer is built using the Hugging Face 🤗 Transformers library, specifically the LLaMA model, which allows for efficient training and generation of high-quality music.

The model is trained using PyTorch Lightning for efficient and scalable training, and Hydra is employed for configuration management.

## Dataset

The model is trained on a combination of the Lakh MIDI Dataset and/or the MAESTRO dataset, providing a diverse collection of musical pieces for learning the structure and patterns in music.

## Features

- Utilizes the powerful LLAMA model from the Hugging Face Transformers library
- Trains the model using PyTorch Lightning for efficient and scalable training
- Employs Hydra for configuration management, making it easy to experiment with different settings
- Supports training on the Lakh MIDI Dataset and the MAESTRO dataset

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/georgi/music-transformer.git
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Prepare your training data:

   ```
   python preprocess.py lmd
   python chunker.py
   ```

2. Train the Music Transformer model:

   ```
   python run.py
   ```

3. Generate new music pieces:
   TBD

4. Enjoy the generated music!

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- [Hugging Face](https://huggingface.co/) for providing the powerful 🤗 Transformers library
- [LLaMA](https://arxiv.org/abs/2302.13971) for the underlying language model architecture
- [Pop Music Transformer](https://arxiv.org/abs/2002.00212)
- [Music Transformer](https://magenta.tensorflow.org/music-transformer)
