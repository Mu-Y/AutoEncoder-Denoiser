import numpy as np 


def dataMix(S, N, fs, mix, TOY = False):
	if TOY:
		start = 60  # start time 60s
		t = 10     # duration 10 seconds
	else:
		start = 30  # start time 30s
		t = 150     # duration 2min 30s
	X = S[fs * start : fs * (start + t)] + mix * N[fs * start : fs * (start + t)]  # mix
	S = S[fs * start : fs * (start + t)]
	# X = X.astype("int16")
	return X, S
def seq2mat(X, Frame_len, shift):
	"""
	sequence to matrix
	"""
	n_Frame = int((len(X) - Frame_len) / shift) + 1
	# print(n_Frame)
	X_mat = np.zeros((n_Frame, Frame_len))   # rfft length will be (n/2 + 1)
	for i in range(n_Frame):
		X_mat[i, :] = X[i * shift : i * shift + Frame_len]

	return X_mat
def time2CompSpectro(X):
	"""
	convert to complex freq value
	"""
	Frame_len = X.shape[1]
	FREQ = np.fft.rfft(X * np.hamming(Frame_len), n = Frame_len)   # keep the magnitude info

	return FREQ

def freq2mel(n_M, fs, FREQ):
	"""
	Generate the F2M matrix
	"""
	half_fs = fs/2
	n_F = FREQ.shape[1]    # dim of freq vector

	M_max = 2595 * np.log10(1 + half_fs / 700)
	m = np.linspace(0, M_max, num = n_M, endpoint = False)
	m = m + M_max / (2 * n_M)   # shift m, now each value in m represents the peak value
	f = (np.power(10.0, m / 2595) - 1) * 700
	f = f/half_fs * (n_F - 1)       # rescale 0-8000 to 0-256

	F2M = np.zeros((n_F, n_M))

	# handle case for i = 0
	fstart = 0
	fmid = f[0]
	fend = f[1]
	# fill in values based on distance to fmid
	for j in range(n_F):
		if j > fstart and j < fend:
			if j < fmid:
				F2M[j, 0] = (j - fstart) / (fmid - fstart)
			else:
				F2M[j, 0] = (fend - j) / (fend - fmid)

	# handle case for 39>i>0
	for i in range(1, n_M - 1):  # i = 1, 2, 3 ... 38 total of 38 terms
		fstart = f[i - 1]
		fmid = f[i]
		fend = f[i + 1]
		# fill in values based on distance to fmid
		for j in range(n_F):
			if j > fstart and j < fend:
				if j < fmid:
					F2M[j, i] = (j - fstart) / (fmid - fstart)
				else:
					F2M[j, i] = (fend - j) / (fend - fmid)

	# handle case for i = 39
	fstart = f[n_M - 2]
	fmid = f[n_M - 1]
	fend = n_F - 1
	# fill in values based on distance to fmid
	for j in range(n_F):
		if j > fstart and j < fend:
			if j < fmid:
				F2M[j, n_M - 1] = (j - fstart) / (fmid - fstart)
			else:
				F2M[j, n_M - 1] = (fend - j) / (fend - fmid)
	# Normalize F2M vertically
	F2M = F2M / np.sum(F2M, axis = 0, keepdims = True)
	return F2M
def splitDataset(X, train_ratio = 0.9, dev_ratio = 0.09, test_ratio = 0.01, seed = None):
	"""
	split training, dev and test set from data matrix
	shuffle the frames
	"""
	m = X.shape[0]    # num of frames
	np.random.seed(seed)
	idx = np.random.permutation(np.arange(m))

	X_train = X[idx[:int(m * train_ratio)], :]
	X_dev = X[idx[int(m * train_ratio) : int(m * train_ratio) + int(m * dev_ratio)], :]
	X_test = X[idx[int(m * train_ratio) + int(m * dev_ratio) : -1], :]

	return X_train, X_dev, X_test

def compspectro2Time(FS_rec, Frame_len, shift):
	"""
	REconstruct. Convert freq signal to time signal. Apply overlap-and-add
	Note FS_rec are complex values
	Arguments:
	----------
	FS_rec: Reconstructed Freq domain signal, complex values, matrix form
	Returns:
	----------
	S_rec: Time domain signal, real values

	"""
	Inv = np.fft.irfft(FS_rec)  # Apply inverse fft to get back matrix-form time signal
	Inv = Inv * np.hamming(Frame_len)
	S_rec = np.zeros(((Inv.shape[0] - 1) * shift + Frame_len, ))
	for i in range(Inv.shape[0]):
		S_rec[i * shift: i * shift + Frame_len] += Inv[i, :] 

	return S_rec

def norm_float2int(S_rec):
    """
    Normalize a float array to int array, for writing wavfile purpose
    Arguments:
    ---------------
    S_rec:  numpy array, float
    Returns:
    ---------------
    S_rec_int:  normalized int  numpy array.
    """
    
    S_rec_mean = np.mean(S_rec)
    S_rec = S_rec - S_rec_mean
    S_rec_int = 32767 * S_rec / np.max(abs(S_rec))
    S_rec_int = S_rec_int.astype("int16")
    
    return S_rec_int

def dataNorm(X):
	"""
	Normalize NN input data 
	"""
	# X_mean = np.mean(X, axis = 0)
	# X_std = np.std(X, axis = 0)

	# X_norm = (X - X_mean) / X_std

	X_norm = X / np.max(np.abs(X))
	return X_norm
