# voice-conversion

### Abstract
The Voice Conversion task involves converting speech from one speaker’s (source) voice to another speaker’s (target) voice. Machine learning methods can be made to perform better than plain signal processing techniques as they can take into account multiple features of speech which cannot be characterized easily by signal processing techniques. In this project, we have explored the use of Recurrent Neural Networks (RNNs) for Voice Conversion. We have explored multiple variations of RNNs using LSTMs and GRUs and observed the effects of changing various parameters of the models. Our approach uses two independently trained neural networks - one which converts source speech to phonemes and another which converts phonemes to target speech. We will present the results achieved by both the networks for these different parameters.

### Datasets
We have made use of the TIMIT dataset which has frame level phoneme transcriptions for utterances by 630 speakers, for training the first neural network. In addition, we’ve used the CMU Arctic dataset for training our second neural network. The Arctic dataset consists of 1150 utterances from a single male and female speaker (target).

### Methodology
We have used a sequence to sequence approach using Recurrent Neural Networks. The architecture is divided into two stages. The first stage (Net1) comprises of converting MFCCs (Mel Frequency Cepstral Coefficients) extracted from the source waveform to phonemes. These are fed into the next neural network (Net2) which converts phonemes to the target waveform.
We have tried different architectures for both the networks, including variations of LSTMs and GRUs. We have explored the effects of changes in the models such as varying the number of hidden layers, dropout rate, creating a pyramidal network structure and doing multitask training. We have trained both the networks individually for these different cases and observed their effect.

Find the full report [here](https://drive.google.com/open?id=1sB5AmQgSCWsvuZm2G48DicbUkRROcpop) and the presentation [here](https://drive.google.com/open?id=13PgDfOSsjoo3o8xv56Xjiw3PiQzfULro).

### Made by
- Arpan Banerjee
- Nihal Singh
- Srivatsan Sridhar