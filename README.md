# speech-emotion-recognition
Project Report: Emotion Recognition

Introduction:
Speech Emotion Recognition (SER) is a field in artificial intelligence that aims to identify the
emotions conveyed in speech signals. In this project, we have built a Speech Emotion Recognition
system using a Long Short-Term Memory (LSTM) neural network and Mel-frequency cepstral
coefficients (MFCC) as features. The dataset used for this project is the "TESS Toronto emotional
speech set data," containing speech samples labeled with different emotions.

Data Loading and Preprocessing:
By downloading the datasets from Kaggle. Then uploaded into the google drive to perform the
program on google colab.
We started by loading the dataset, which contains audio files and their corresponding emotion labels.
paths = []
labels = []
for dirname, _, filenames in os.walk('/content/drive/MyDrive/TESS Toronto emotional speech set
data'):
for filename in filenames:
paths.append(os.path.join(dirname, filename))
label = filename.split('_')[-1]
label = label.split('.')[0]
labels.append(label.lower())
if len(paths) == 2800:
break
print('Dataset is Loaded')
The dataset is divided into seven emotion categories: fear, angry, disgust, neutral, sad, ps (pleasant
surprise), and happy. We created a Pandas Data Frame to store the file paths and emotion labels for
each audio sample.

Exploratory Data Analysis:
To gain insights into the dataset, we performed exploratory data analysis. We checked the distribution of
samples across different emotion categories to ensure that the dataset is balanced.
sns.countplot(x='label', data=df)
plt.Ɵtle('DistribuƟon of EmoƟon Labels')
plt.show()

Audio and Spectrogram Visualization:
Next, we visualized the waveforms and spectrograms of audio samples for each emotion category. The
wave plot displays the audio waveform, and the spectrogram represents the frequency content of the
audio signal over time. These visualizations provide a better understanding of the audio data.
def waveplot(data, sr, emotion):
plt.figure(figsize=(10, 4))
plt.title(emotion, size=20)
librosa.display.waveshow(data, sr=sr)
plt.show()
def spectrogram(data, sr, emotion):
plt.figure(figsize=(10, 4))
plt.title(emotion, size=20)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(data)), ref=np.max), sr=sr,
x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.show()
emotions_to_visualize = ['fear', 'angry', 'disgust', 'neutral', 'sad', 'ps', 'happy']
for emotion in emotions_to_visualize:
path = np.array(df['speech'][df['label'] == emotion])[0]
data, sampling_rate = librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectrogram(data, sampling_rate, emotion)
Feature Extraction:
For training our LSTM model, we extracted Mel-frequency cepstral coefficients (MFCC) from the audio
samples. MFCCs are commonly used as features for speech-related tasks due to their effectiveness in
capturing speech characteristics.
def extract_mfcc(filename):
y, sr = librosa.load(filename, duration=3, offset=0.5)
mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
return mfcc
X_mfcc = df['speech'].apply(lambda x: extract_mfcc(x))
X = [x for x in X_mfcc]
X = np.array(X)
LSTM Model Architecture:
The LSTM model consists of an LSTM layer with 256 units followed by Dropout layers to prevent
overfitting. We added Dense layers with RELU activation functions to process the extracted MFCC
features. The final Dense layer has 7 units (equal to the number of emotion categories) with a soft max
activation function for multiclass classification.
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
model = Sequential([
LSTM(256, return_sequences=False, input_shape=(40,1)),
Dropout(0.2),
Dense(128, activation='relu'),

Dropout(0.2),
Dense(64, activation='relu'),
Dropout(0.2),
Dense(7, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
history = model.fit(X, y, validation_split=0.2, epochs=50, batch_size=64)

Model Training and Evaluation:
We compiled the LSTM model with categorical cross-entropy loss and the Adam optimizer. The model was
trained on the extracted MFCC features with a validation split of 20% and 50 epochs. The training progress was
monitored using accuracy and loss metrics.
epochs = list(range(50))
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, label='train accuracy')
plt.plot(epochs, val_acc, label='val accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(epochs, loss, label='train loss')
plt.plot(epochs, val_loss, label='val loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

Results and Performance:
After training the LSTM model, we visualized the training and validation accuracy as well as the
training and validation loss over the 50 epochs. These plots help us understand the model's performance
and identify potential overfitting or underfitting issues.

Conclusion:
In this project, we successfully developed a Speech Emotion Recognition system using an LSTM neural
network and MFCC features. The trained model can predict the emotions associated with speech audio with a
reasonable accuracy rate. Further improvements can be made by experimenting with different model
architectures, feature extraction techniques, and hyperparameter tuning.
