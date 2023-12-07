import os
import librosa
from mutagen.id3 import ID3, TIT2, TALB, TPE1, TBPM, TKEY
from mutagen.id3 import ID3NoHeaderError
from PyQt5 import QtWidgets, QtGui, QtCore
import shutil
import numpy as np

class MyWindow(QtWidgets.QWidget):
    # Class-level dictionary to store BPM and key information
    track_info = {}

    def __init__(self):
        super(MyWindow, self).__init__()

        self.setWindowTitle("BPM & Key Tagger")
        self.setGeometry(100, 100, 400, 200)

        layout = QtWidgets.QVBoxLayout()

        self.input_label = QtWidgets.QLabel("Input Directory:")
        self.input_entry = QtWidgets.QLineEdit()
        self.browse_input_button = QtWidgets.QPushButton("Browse")
        self.browse_input_button.clicked.connect(self.browse_input_directory)

        self.output_label = QtWidgets.QLabel("Output Directory:")
        self.output_entry = QtWidgets.QLineEdit()
        self.browse_output_button = QtWidgets.QPushButton("Browse")
        self.browse_output_button.clicked.connect(self.browse_output_directory)

        self.analyze_button = QtWidgets.QPushButton("Analyze")
        self.analyze_button.clicked.connect(self.analyze)

        self.write_button = QtWidgets.QPushButton("Write")
        self.write_button.clicked.connect(self.write_tags)

        self.progress_analyze = QtWidgets.QProgressBar()
        self.progress_write = QtWidgets.QProgressBar()

        layout.addWidget(self.input_label)
        layout.addWidget(self.input_entry)
        layout.addWidget(self.browse_input_button)
        layout.addWidget(self.output_label)
        layout.addWidget(self.output_entry)
        layout.addWidget(self.browse_output_button)
        layout.addWidget(self.analyze_button)
        layout.addWidget(self.write_button)
        layout.addWidget(self.progress_analyze)
        layout.addWidget(self.progress_write)

        self.setLayout(layout)

    def browse_input_directory(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Input Directory")
        if directory:
            self.input_entry.setText(directory)

    def browse_output_directory(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_entry.setText(directory)

    def analyze_track(self, track_path):
        # Load audio file
        y, sr = librosa.load(track_path, duration=10)  # Analyzing only the first 10 seconds

        # Tempo and beat tracking
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

        # Tonnetz analysis
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        tonnetz = librosa.feature.tonnetz(y=y_harmonic)

        # Mock data for demonstration
        bpm = round(tempo)
        key = self.key_from_tonnetz(tonnetz)

        return bpm, key

    def analyze(self):
        input_directory = self.input_entry.text()

        if not os.path.exists(input_directory):
            print("Invalid input directory.")
            return

        input_files = [f for f in os.listdir(input_directory) if f.endswith(".mp3")]

        self.progress_analyze.setRange(0, len(input_files))
        self.progress_analyze.setValue(0)

        for i, input_file in enumerate(input_files):
            input_path = os.path.join(input_directory, input_file)
            bpm, key = self.analyze_track(input_path)

            # Store BPM and key information in the class-level dictionary
            self.track_info[input_file] = {'bpm': bpm, 'key': key}

            # Update progress bar
            self.progress_analyze.setValue(i + 1)
            QtWidgets.QApplication.processEvents()

        print("Analysis completed.")

    def key_from_tonnetz(self, tonnetz):
        # Sumowanie tonnetz wzdłuż osi czasu
        sum_tonnetz = np.sum(tonnetz, axis=1)

        # Indeks największej sumy odpowiada dominującemu kluczowi
        key_index = np.argmax(sum_tonnetz)

        # Przeliczenie indeksu na wartość MIDI
        midi_value = (key_index + 3) % 12  # Dźwięki MIDI zaczynają się od C (indeks 0), stąd +3

        # Mapowanie wartości MIDI na nazwę klucza
        key_values = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        key = key_values[midi_value]

        return key


    def get_key_from_index(self, index):
        # Map the index to the corresponding key value
        key_values = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        key_index = int(index) % len(key_values)
        return key_values[key_index]

    def write_tags(self):
        output_directory = self.output_entry.text()

        if not os.path.exists(output_directory):
            print("Invalid output directory.")
            return

        input_directory = self.input_entry.text()

        if not os.path.exists(input_directory):
            print("Invalid input directory.")
            return

        # Backup the entire input directory to the output directory
        backup_directory = os.path.join(output_directory, "updated")
        shutil.copytree(input_directory, backup_directory)

        input_files = [f for f in os.listdir(input_directory) if f.endswith(".mp3")]

        self.progress_write.setRange(0, len(input_files))
        self.progress_write.setValue(0)

        for i, input_file in enumerate(input_files):
            input_path = os.path.join(input_directory, input_file)
            output_path = os.path.join(output_directory, "updated", input_file)

            if not os.path.exists(input_path):
                print(f"File not found: {input_path}")
                continue

            # Get BPM and key from the stored information
            bpm = self.track_info[input_file]['bpm']
            key = self.track_info[input_file]['key']

            try:
                audiofile = ID3(output_path)
            except ID3NoHeaderError:
                # Create a new ID3 tag if the file doesn't have one
                audiofile = ID3()

            # Write BPM
            bpm_frame = TBPM(encoding=3, text=str(bpm))
            audiofile['TBPM'] = bpm_frame

            # Write Key
            key_frame = TKEY(encoding=3, text=key)
            audiofile['TKEY'] = key_frame

            # Save tags to the destination file
            audiofile.save(output_path)

            # Update progress bar
            self.progress_write.setValue(i + 1)
            QtWidgets.QApplication.processEvents()

        print("Write completed.")

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    win = MyWindow()
    win.show()
    app.exec_()

