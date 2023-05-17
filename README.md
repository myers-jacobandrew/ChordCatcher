# ChordCatcher

ChordCatcher is a project aimed at automatically recognizing and transcribing guitar chords from video and audio recordings. The project uses computer vision and machine learning techniques to analyze guitar playing footage and identify the chords being played.

## Getting Started

To get started with the project, you will need to clone this repository to your local machine. Once you have done that, you can follow the instructions below to set up the project environment and start running the code.

### Prerequisites

To run this project, you will need to install the following dependencies:

    Python 3.x
    TensorFlow
    NumPy
    SciPy
    Scikit-learn
    Librosa

You can install these dependencies by running pip install -r requirements.txt.


### Usage

To use the ChordCatcher project, run the following command:

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
  This project was inspired by the work of Qingyang Xi, Rachel Bittner, Xuzhou Ye, and Juan Pablo Bello from NYU's Music and Audio Research Lab, as well as Johan Pauwels from the Center for Digital Music at Queen Mary University of London, who created the GuitarSet dataset used in this project.  More information can be found at https://guitarset.weebly.com/.
  
  The VGGish model used in this project was developed by Shawn Hershey, Sourish Chaudhuri, Daniel PW Ellis, Jort F Gemmeke, Aren Jansen, and Channing Moore, and is available under the Apache License 2.0.
  
  The TensorFlow library was used to implement the machine learning algorithms in this project.
  
  Anticipating the use of Aayush Kandpal collection of Guitar_Chords_Finger_positions found [here](https://www.kaggle.com/datasets/aayushkandpal/guitar-chords-finger-positions?select=chord-fingers.csv).

