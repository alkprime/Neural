# Neural
 Start from Main
 
 This is an educational neural network
 build using pointers from leNet original network, and using their labled images
 It is my take on a neural network build from the scratch, not using tensorflow or pytorch.
 
 written as a mix of CNN and flat
 for learning purposes of different layer sizes
 uses:
 - numpy, data manipulation.
 - idx2numpy, covert original image & label data to numpy readable arrays.
 - matplotlib, mainly for cost plotting over iterations.
 - tqdm, cmd progress bar plotting, waiting without visual reference was getting on my nerves.
  
 pay attention to the imported libraries.
 
 to do:
 - better numerical labeling of neural layers & hyperparameters.
 - finish the CNN (almost done, some fixes)
 - add a GUI.
 - write a better read me.

 Progress:
 - CNN forward propagation is working.
 - CNN backprop is work in progress
 - Pooling layer takes Z and returns an A, check for better solution
 - put activation function after the pooling layer (computation saving)