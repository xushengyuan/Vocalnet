# Vocalnet
A wavenet and waveglow based singing synthesizing system.

## Versions
There are two different versions of the Vocalnet singing synthesizing system.
- Vocalnet e2e
- Vocalnet mdf

The e2e means end-to-end, it only employs a conditional wavenet to generate mel-spectrum from a one-hot encoded midi like input from the user,
 and a waveglow vocoder to systhesize waveform from the mel-spectrum at speed faster than realtime. 
 To train a model for it is very easy, because the dataset only need a midi like input and taret waveform. It doesn't need any detail labels.
 
 The mdf means modify, it uses a more complex but more flexible architecture, it employs multiple wavenets to generate phonemes, loudness and pitch.
 Then use a main wavenet model to generate the final mel-spectrum for the waveglow vocoder as mentioned above. This architecture enables users
 to manually modify vocal features or make a combination of different models to get a unique characteristic for their songs.
