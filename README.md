# Vocalnet
A wavenet and waveglow based singing synthesizing system

The e2e means end-to-end, it only employs a conditional wavenet to generate mel-spectrum from a one-hot encoded midi-like input from the user,
 and a waveglow vocoder to systhesize waveform from the mel-spectrum at speed faster than realtime. 
 To train a model for it is very easy, because the dataset only need a midi like input and taret waveform. It doesn't need any detail labels.
