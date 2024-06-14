# semg2vec: Learning vector representations from electromyography for improved gesture recognition and biometric identification

Surface electromyography (sEMG) signals are measured at the arm in a non-invasive way and correspond to muscle activity. They have been used for many health monitoring applications, biometric identification and prosthetics control. The research in this field mainly concentrates on training models on only one classification.
	
In this thesis a new approach is tested. A meaningful vector representation is constructed for sEMG signals. This approach can help to build a model that is more generalizable. This will lead to a robust model which will be able to classify unknown gestures better and can also be used for different classification tasks. Such a model would be beneficial for controlling prosthetics hands, since gesture recognition is the key to execute certain gestures with prosthetic hands.  
	
To build a meaningful vector representation, possible approaches are first analysed. Afterwards two models are implemented. The first approach is called _semg2vec_ and is inspired by the famous _word2vec_. The second approach is a state-of-the-art vision transformer. The trained representation of both models are tested on new gestures and biometric identification. Additionally, the gesture recognition results of the models are compared to a simple multi-layer perceptron.
	
The vision transformer shows better results than _semg2vec_. It performed better on gesture recognition and was able to classify new gestures better. The biometric identification revealed a probable data leakage and a systematic distortion in the dataset. The multi-layer perceptron performed better on simple gesture recognition. That shows, that a simpler model is more suitable for a single task. 
	
Overall, the vision transformer is a promising model to build a meaningful vector representation for sEMG signals. With further tuning and a bigger dataset the model could be able to extract more meaningful data to better classify unknown gestures and can also be expanded to more classification tasks like biometric identification.

The code is based on https://github.com/MansoorehMontazerin/Vision-Transformer_EMG
