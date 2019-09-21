import os
import subprocess
from scipy.io import wavfile
from scipy.signal import spectrogram
import numpy as np
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.initializers import glorot_normal
from keras.layers import Input, Dense, BatchNormalization, Dropout
from keras.models import Model, load_model
from keras import optimizers
from keras.utils  import plot_model
from time import gmtime, strftime
from utils import dataMix, time2CompSpectro, freq2mel, seq2mat, dataNorm, norm_float2int, compspectro2Time, splitDataset
import argparse
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


URL_S =  "https://www.youtube.com/watch?v=oDJ0dRb4-XY"
# URL_N = "https://www.youtube.com/watch?v=LS3O6QBcaxs&t="
URL_N =   "https://www.youtube.com/watch?v=4t5_LbJBrRU"  # Cafeteria noise
# URL_N = "https://www.youtube.com/watch?v=j9nhecEWMuE"  # raining noise
# URL_S_listen = "https://www.youtube.com/watch?v=MhOdbtPhbLU"
URL_S_listen = "https://www.youtube.com/watch?v=7sxpKhIbr0E"
# URL_N_listen =  "https://www.youtube.com/watch?v=TpdFVSi7PZ8"
# URL_N_listen = "https://www.youtube.com/watch?v=U2-eC8z-J5 g"  # cafe noise
URL_N_listen = "https://www.youtube.com/watch?v=j9nhecEWMuE" # raining noise


Frame_len = 512
shift = 128
mix = 0.7
n_Mel = 40
dr_rate = 0.3

if __name__ == '__main__':
	p = argparse.ArgumentParser()
	p.add_argument('--DOWNLOAD', action='store_true')
	p.add_argument('--TOY', action='store_true')
	p.add_argument('--TRAIN', action='store_true')
	p.add_argument('--RECONSTRUCT', action='store_true')
	p.add_argument('--PLOT', action='store_true')

	args = p.parse_args()


	if args.DOWNLOAD:   
		# download audio 
		subprocess.run(["youtube-dl", "-x", "--audio-format", "wav", "-o", "trainspeech_raw.%(wav)s", URL_S])
		subprocess.run(["youtube-dl", "-x", "--audio-format", "wav", "-o", "trainnoise_raw.%(wav)s", URL_N])
		subprocess.run(["youtube-dl", "-x", "--audio-format", "wav", "-o", "testspeech_raw.%(wav)s", URL_S_listen])
		subprocess.run(["youtube-dl", "-x", "--audio-format", "wav", "-o", "testnoise_raw.%(wav)s", URL_N_listen])
		#preprocess audio, including resampling, single-channelizing
		subprocess.run(["sox", "--norm", "trainspeech_raw.wav", "-c1", "--norm", "trainspeech.wav", "--norm", "rate", "16000"])
		subprocess.run(["sox", "--norm", "trainnoise_raw.wav", "-c1", "--norm", "trainnoise.wav", "--norm", "rate", "16000"])
		subprocess.run(["sox", "--norm", "testspeech_raw.wav", "-c1", "--norm", "testspeech.wav", "--norm","rate", "16000"])
		subprocess.run(["sox", "--norm", "testnoise_raw.wav", "-c1", "--norm", "testnoise.wav", "--norm", "rate", "16000"])



	# generate raw data from wav file
	fs, S = wavfile.read("trainspeech.wav")
	fs, N = wavfile.read("trainnoise.wav")


	S  = norm_float2int(S)
	N  = norm_float2int(N)


	# mix noise and speech, trim 10 seconds if TOY, trim 2min 30seconds otherwise
	X, S = dataMix(S, N, fs, mix, TOY = args.TOY) 


	X_int = norm_float2int(X)
	wavfile.write("X_train.wav", fs, X_int)
	S_int = norm_float2int(S)
	wavfile.write("S_train.wav", fs, S_int)

	S_mat = seq2mat(S, Frame_len, shift)
	X_mat = seq2mat(X, Frame_len, shift)
	S_train, S_dev, S_test = splitDataset(S_mat, seed = 1)
	X_train, X_dev, X_test = splitDataset(X_mat, seed = 1)

	FS_train = time2CompSpectro(S_train)
	FX_train = time2CompSpectro(X_train)
	FS_dev = time2CompSpectro(S_dev)
	FX_dev = time2CompSpectro(X_dev)

	F2M = freq2mel(n_Mel, fs, FX_train)
	# get M2F matrix by transpose
	M2F = F2M.T
	# normalize M2F matrix, plus a very small number to avoid underflow
	M2F = M2F / (np.sum(M2F, axis = 0, keepdims = True) + 0.00000001)

	MEL_S_train = np.dot(np.abs(FS_train), F2M)  # output of NN (MEL_S/MEL_X)
	MEL_X_train = np.dot(np.abs(FX_train), F2M)  # input of NN (MEL_X)

	MEL_S_dev = np.dot(np.abs(FS_dev), F2M)
	MEL_X_dev = np.dot(np.abs(FX_dev), F2M)
	######## Data normalization ##########
	MEL_S_train_norm = dataNorm(MEL_S_train)
	MEL_X_train_norm = dataNorm(MEL_X_train)
	MEL_S_dev_norm = dataNorm(MEL_S_dev)
	MEL_X_dev_norm = dataNorm(MEL_X_dev)
	######################################


	if args.TRAIN:
		n_hidden = [200, 300, 200]

		inputs = Input(shape = (n_Mel, ))

		x = BatchNormalization()(inputs)
		x = Dropout(rate = dr_rate)(x)

		x = Dense(n_hidden[0], activation = "relu")(x)
		x = BatchNormalization()(x)
		x = Dropout(rate = dr_rate)(x)

		x = Dense(n_hidden[1], activation = "relu")(x)

		x = Dense(n_hidden[2], activation = "relu")(x)
		x = BatchNormalization()(x)
		x = Dropout(rate = dr_rate)(x)

		outputs = Dense(40, activation = "relu")(x)

		model = Model(inputs = inputs, outputs = outputs)

		# plot_model(model, to_file = "model.png", show_shapes = True, show_layer_names = True)
		# model.summary()
		tensorboard = TensorBoard(log_dir="logs/"+strftime("%Y_%m_%d-%H_%M_%S", gmtime()) , histogram_freq=0, batch_size=32, \
								write_graph=True, write_grads=False, write_images=False, \
								embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None  )

		adam = optimizers.Adam(lr = 0.01)
		model.compile(optimizer = adam, loss = "mean_squared_error")
		model.fit(x = MEL_X_train_norm, y = MEL_S_train_norm / MEL_X_train_norm, epochs = 30, verbose = 2, callbacks = [tensorboard], \
					validation_data = (MEL_X_dev_norm, MEL_S_dev_norm / MEL_X_dev_norm))
		model.save("model.h5")


	if args.RECONSTRUCT:
		model = load_model("model.h5")

		##################################################################
		########### Reconstruct denoised Training speech ################
		FX = time2CompSpectro(X_mat)
		MEL_X = np.dot(np.abs(FX), F2M)
		MEL_X_norm = dataNorm(MEL_X)

		MEL_gain = model.predict(MEL_X_norm)
		F_gain = np.dot(MEL_gain, M2F)

		FD = F_gain * FX 
		######### reconstruct time domain signal ################
		D_rec = compspectro2Time(FD, Frame_len, shift)    
		D_rec_int = norm_float2int(D_rec)
		wavfile.write("D_train.wav", fs, D_rec_int)

		# f, t, Sxx = spectrogram(X[0:480000], fs, nperseg = 128)
		# plt.pcolormesh(t, f, Sxx)
		# plt.ylabel('Frequency [Hz]')
		# plt.xlabel('Time [sec]')
		# plt.savefig('X_train_spectro.png')
		# plt.close()

		if args.PLOT:
			plt.figure(figsize = (10, 7))
			plt.plot(X, label = 'Mix speech (Training)')
			plt.plot(S, label = 'Clean speech (Training)')
			plt.xlabel("Sample")
			plt.ylabel("Magnitude")
			plt.legend()
			plt.title("Mix speech vs. Clean speech (Training)")
			plt.savefig("Mix speech vs. Clean speech (Training).png")
			plt.close()

			plt.figure(figsize = (10, 7))
			plt.plot(X, label = 'Mix speech (Training)')
			plt.plot(D_rec, label = 'Denoised speech (Training)')
			plt.xlabel("Sample")
			plt.ylabel("Magnitude")
			plt.legend()
			plt.title("Mix speech vs. Denoised speech (Training)")
			plt.savefig("Mix speech vs. Denosied speech (Training).png")
			plt.close()

			plt.figure(figsize = (10, 7))
			plt.plot(D_rec, label = 'Denoised speech (Training)')
			plt.plot(S, label = 'Clean speech (Training)')
			plt.xlabel("Sample")
			plt.ylabel("Magnitude")
			plt.legend()
			plt.title("Denoised speech vs. Clean speech (Training)")
			plt.savefig("Denoised speech vs. Clean speech (Training).png")
			plt.close()

		
		##################################################################
		##################################################################


		##################################################################
		########### Reconstruct denoised Test speech ################
		fs, S = wavfile.read("testspeech.wav")
		fs, N = wavfile.read("testnoise.wav")


		S  = norm_float2int(S)
		N  = norm_float2int(N)
		# mix noise and speech, trim 10 seconds if TOY, trim 2min 30seconds otherwise
		X, S = dataMix(S, N, fs, mix, TOY = args.TOY)  

		X_int = norm_float2int(X)
		wavfile.write("X_test.wav", fs, X_int)
		S_int = norm_float2int(S)
		wavfile.write("S_test.wav", fs, S_int)



		S_mat = seq2mat(S, Frame_len, shift)
		X_mat = seq2mat(X, Frame_len, shift)

		FX = time2CompSpectro(X_mat)
		MEL_X = np.dot(np.abs(FX), F2M)
		MEL_X_norm = dataNorm(MEL_X)

		MEL_gain = model.predict(MEL_X_norm)
		F_gain = np.dot(MEL_gain, M2F)

		FD = F_gain * FX 
		######### reconstruct time domain signal ################
		D_rec = compspectro2Time(FD, Frame_len, shift) 
		D_rec_int = norm_float2int(D_rec)
		wavfile.write("D_test.wav", fs, D_rec_int)

		if args.PLOT: 
			plt.figure(figsize = (10, 7))
			plt.plot(X, label = 'Mix speech (Test)')
			plt.plot(S, label = 'Clean speech (Test)')
			plt.xlabel("Sample")
			plt.ylabel("Magnitude")
			plt.legend()
			plt.title("Mix speech vs. Clean speech (Test)")
			plt.savefig("Mix speech vs. Clean speech (Test).png")
			plt.close()

			plt.figure(figsize = (10, 7))
			plt.plot(X, label = 'Mix speech (Test)')
			plt.plot(D_rec, label = 'Denoised speech (Test)')
			plt.xlabel("Sample")
			plt.ylabel("Magnitude")
			plt.legend()
			plt.title("Mix speech vs. Denoised speech (Test)")
			plt.savefig("Mix speech vs. Denosied speech (Test).png")
			plt.close()

			plt.figure(figsize = (10, 7))
			plt.plot(D_rec, label = 'Denoised speech (Test)')
			plt.plot(S, label = 'Clean speech (Test)')
			plt.xlabel("Sample")
			plt.ylabel("Magnitude")
			plt.legend()
			plt.title("Denoised speech vs. Clean speech (Test)")
			plt.savefig("Denoised speech vs. Clean speech (Test).png")
			plt.close()


		

		##################################################################
		##################################################################






