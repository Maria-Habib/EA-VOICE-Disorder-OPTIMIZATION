import librosa, sklearn
import numpy as np
import warnings
import pandas as pd
import wfdb
pd.set_option("display.max_rows", 999)


warnings.filterwarnings('ignore')



df = pd.read_csv('svd_dataset.tsv', sep='\t')

paths = df['path'].values.tolist()
labels = df['label'].values.tolist()
#path = '4PK1A.wav'



def extract_mfcc(x, sr):
    # the function take a signal (x) and the sampling rate (sr) and return the avg and std of mfcc
    mfcc = librosa.feature.mfcc(x, sr=sr, n_mfcc=40)
    mfcc = sklearn.preprocessing.scale(mfcc, axis=1)

    mn = np.mean(mfcc.T, axis=0)
    std = np.var(mfcc.T, axis=0)

    mfcc_header = [*[f'mffc_mn_{i}' for i in range(40)], *[f'mffc_sd_{i}' for i in range(40)]]
    mf = [*mn, *std]

    return mf, mfcc_header



def extract_melspectrogram(x, sr):
    mel_spec = librosa.feature.melspectrogram(y=x, sr=sr, n_fft=2048, hop_length=1024)
    mel_spect = librosa.power_to_db(mel_spec, ref=np.max)
    # average 
    
    mel_spect = np.mean(mel_spect, axis=1).T
    mspec_header = [f'mel_spec_{i}' for i in range(len(mel_spect))]
    return mel_spect, mspec_header




def extract_rythm(x, sr):
    # Compute the beats and tempo feature
    hop_length = 512

    onset_env = librosa.onset.onset_strength(x, sr=sr)
    dtempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, aggregate=None)
    _, beats = librosa.beat.beat_track(y=x, sr=sr)

    # Compiute the fourier tempogram
    tempogram = librosa.feature.fourier_tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop_length)

    # Compute the auto-correlation tempogram, unnormalized to make comparison easier
    ac_tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop_length, norm=None)  

    #print(len(dtempo), type(dtempo))
    #print(len(beats), type(beats)) beats are many times zero, I'll exclude them
    #print(len(tempogram), type(tempogram))
    #print(len(ac_tempogram), type(ac_tempogram))

    tempogram = list(np.mean(tempogram.astype(np.float32).tolist(), axis=1))
    ac_tempogram = list(np.mean(ac_tempogram.astype(np.float32).tolist(), axis=1))
    
    results = tempogram + ac_tempogram + [dtempo.mean(axis=-1)]
    res_header = [f'fft_tempo_{i}' for i in range(len(tempogram))] + [f'corr_tempo_{i}' for i in range(len(ac_tempogram))] + ['dynm_tempo_mn']


    return results, res_header




def extract_spectral(x, sr):
    # Compute the chroma
    chroma_mn = np.mean(librosa.feature.chroma_stft(y=x, sr=sr).tolist(), axis=1).tolist()
    chroma_sd = np.std(librosa.feature.chroma_stft(y=x, sr=sr).tolist(), axis=1).tolist()
    #print(chroma_mn, chroma_sd)


    # Compute root-mean-square (RMS) value for each frame
    rms_mean = np.mean(librosa.feature.rms(y=x).tolist()[0])
    rms_std = np.std(librosa.feature.rms(y=x).tolist()[0])    
    #print(rms_mean, rms_std)

    # Compute the spectral centroid
    spec_cent = librosa.feature.spectral_centroid(y=x, sr=sr).tolist()[0]
    spec_cent_mn = np.mean(spec_cent)
    spec_cent_sd = np.std(spec_cent)
    #print(spec_cent_mn, spec_cent_sd)

    # Compute the spectral bandwidth
    spec_bw = librosa.feature.spectral_bandwidth(y=x, sr=sr).tolist()[0]
    spec_bw_mn = np.mean(spec_bw)
    spec_bw_sd = np.std(spec_bw)
    #print(spec_bw_mn, spec_bw_sd)

    # Compute the spectral contrast
    S = np.abs(librosa.stft(x))
    contrast = np.mean(librosa.feature.spectral_contrast(S=S, sr=sr), axis=1).tolist()
    #print(contrast)  
  
    # Compute the spectral flatness
    flatness_mn = np.mean(librosa.feature.spectral_flatness(y=x), axis=1)[0]
    flatness_sd = np.std(librosa.feature.spectral_flatness(y=x), axis=1)[0] 
    #print(flatness_mn, flatness_sd)
    
    # Compute roll-off frequency
    rolloff_mn = np.mean(librosa.feature.spectral_rolloff(y=x, sr=sr), axis=1)[0]
    rolloff_sd = np.std(librosa.feature.spectral_rolloff(y=x, sr=sr), axis=1)[0]
    #print(rolloff_mn, rolloff_sd)
    
    # Compute the poly-features at order 0, 1, and 2
    p0_mn = np.mean(librosa.feature.poly_features(S=S, order=0), axis=1)[0]
    p0_sd = np.std(librosa.feature.poly_features(S=S, order=0), axis=1)[0]
    #print(p0_mn, p0_sd)

    p1_mn = np.mean(librosa.feature.poly_features(S=S, order=1), axis=1).tolist()
    p1_sd = np.std(librosa.feature.poly_features(S=S, order=1), axis=1).tolist()
    #print(p1_mn, p1_sd)

    p2_mn = np.mean(librosa.feature.poly_features(S=S, order=2), axis=1).tolist()
    p2_sd = np.std(librosa.feature.poly_features(S=S, order=2), axis=1).tolist()
    #print(p2_mn, p2_sd)

    # Compute the tonnetz from the harmonic component
    y = librosa.effects.harmonic(x)
    tonnetz = np.mean(librosa.feature.tonnetz(y=y, sr=sr), axis=1).tolist()
    #print(len(tonnetz))

    # Compute the zer-crossing rate of time-series
    zcr = librosa.feature.zero_crossing_rate(x).tolist()[0]
    zcr_mn = np.mean(zcr)
    zcr_sd = np.std(zcr)
    #print(zcr_mn, zcr_sd)

    results = chroma_mn + chroma_sd + [rms_mean, rms_std] + [spec_cent_mn, spec_cent_sd] + [spec_bw_mn, spec_bw_sd] + contrast + [flatness_mn, flatness_sd] + [rolloff_mn, rolloff_sd] + [p0_mn, p0_sd] + p1_mn + p1_sd + p2_mn + p2_sd + tonnetz + [zcr_mn, zcr_sd]


    head_chroma = [f'chroma_mn_{i}' for i in range(len(chroma_mn))] + [f'chroma_sd_{i}' for i in range(len(chroma_sd))]
    head_contrast = [f'contrast_{i}' for i in range(len(contrast))]
    head_polly = ['p0_mn', 'p0_sd'] + [f'p1_mn_{i}' for i in range(len(p1_mn))] + [f'p1_sd_{i}' for i in range(len(p1_sd))] + [f'p2_mn_{i}' for i in range(len(p2_mn))] + [f'p2_sd_{i}' for i in range(len(p2_sd))]
    head_tonez = [f'tonnetz_{i}' for i in range(len(tonnetz))] 

    res_header = head_chroma + ['rms_mean', 'rms_std'] + ['spec_cent_mn', 'spec_cent_sd'] + ['spec_bw_mn', 'spec_bw_sd'] + head_contrast + ['flatness_mn', 'flatness_sd'] + ['rolloff_mn', 'rolloff_sd'] + head_polly + head_tonez + ['zcr_mn', 'zcr_sd']


    return results, res_header




mfccs = []
mspec = []
rythm = []
spectral = []
cols = []
print(len(paths))
c = 0


for p in paths:
    c += 1
    print('Extracting features from audio %s' %c)
    
    #x, sr = librosa.load(p)

    record = wfdb.rdrecord(p)
    #print(record)
    x, fields = wfdb.rdsamp(p)
    sr = 16000

    mfcc, h1 = extract_mfcc(np.squeeze(x), sr)
    mfccs.append(mfcc)    
    

    spec, h2 = extract_melspectrogram(np.squeeze(x), sr)
    mspec.append(spec)


    x = np.squeeze(x)

    ryth, h3 = extract_rythm(x, sr)
    rythm.append(ryth)

    spc, h4 = extract_spectral(x, sr)
    spectral.append(spc)

    #print(len(h1), len(h2), len(h3))#, len(h4))
    cols = h1 + h2 + h3 + h4


print(mspec)
print(len(mspec))

df['mfcc'] = mfccs
df['mels'] = mspec
df['rythm'] = rythm
df['spec'] = spectral
   
df_mfcc = pd.DataFrame(df['mfcc'].values.tolist())
df_mels = pd.DataFrame(df['mels'].values.tolist())
df_ryth = pd.DataFrame(df['rythm'].values.tolist())
df_spec = pd.DataFrame(df['spec'].values.tolist())


df = df.drop('mfcc', axis=1)
df = df.drop('mels', axis=1)
df = df.drop('rythm', axis=1)
df = df.drop('spec', axis=1)

print(len(df.columns))
#print(len(df_mfcc.columns))
#print(len(df_mels.columns))
#print(len(df_ryth.columns))
#print(len(df_spec.columns))
print(len(cols))

df_lst = pd.concat([df, df_mfcc, df_mels, df_ryth, df_spec], axis=1)#df_spec
print(len(df_lst.columns))

columns = ['path', 'label'] + cols
#dff = pd.DataFrame(df_lst, index=False)
df_lst.columns = columns


df_lst.to_csv('svd_handcrafted_data.tsv', sep='\t', index=False)



#pd.DataFrame(l).to_csv('voiced_dataset.tsv',iprint(df_lst.shape)
print(df_lst.head())

