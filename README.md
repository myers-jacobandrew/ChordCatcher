# Chord Catcher

Chord Catcher is a project aimed at automatically recognizing and transcribing guitar chords from video and audio recordings. The project uses computer vision and machine learning techniques to analyze guitar playing footage and identify the chords being played.

## Getting Started

To get started with the project, you will need to clone this repository to your local machine. Once you have done that, you can follow the instructions below to set up the project environment and start running the code.

### Prerequisites

The following software should be installed on your local machine:

- Python 3.x
- Anaconda or Miniconda

### Installation

1. Create a new conda environment for the project:

```bash
conda create --name chord_catcher python=3.8
conda activate chord_catcher
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

### Usage

To use the Chord Catcher project, run the following command:

```bash
python main.py
```

This will start the chord recognition process on the sample dataset included in the project.

## Contributing

Contributions to the project are welcome! To contribute, follow the instructions below:

1. Fork this repository.
2. Create a new branch for your changes.
3. Make your changes and commit them to your branch.
4. Push your branch to your fork of the repository.
5. Create a pull request from your branch to the main repository.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

This project was inspired by the GuitarSet dataset from Zenodo, and uses the VGGish feature extraction model developed by Google.

