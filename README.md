# Adversarial Autoencoder

User Interface will have those "steps":

    1. load — images or tabular data 
    2. build — image of the network which interactively builds
    3. train — image of network, with input and output from auto-encoder at x number of iterations 
    plus you see the real distribution updates over time. Then of course show the loss function graphs 
    in real-time.
    4. visualize — image of network, with ability to draw samples to generate, and to see strip of 
    images or observations that you can select in order to see the decoder
    5. tune — setting up grid search, and returning a list of panels, and on each panel you have 
    parameters with over all loss

## MNIST: 

### Unsupervised Autoencoder
<figure align="center">
  <img src="https://github.com/Explainable-Artificial-Intelligence/AdversarialAutoencoder/blob/master/doc/results/mnist/mnist_unsupervised_z_dim_2_reconstruction_grid.png?raw=Truee" alt="my alt text"/>
  <figcaption>This is my caption text.</figcaption>
</figure>

<figure align="center">
  <img src="https://github.com/Explainable-Artificial-Intelligence/AdversarialAutoencoder/blob/master/doc/results/mnist/z_dim_2_50_real_dist_and_encoder_dist.png?raw=Truee" alt="my alt text"/>
  <figcaption>This is my caption text.</figcaption>
</figure>

<figure align="center">
  <img src="https://github.com/Explainable-Artificial-Intelligence/AdversarialAutoencoder/blob/master/doc/results/mnist/mnist_unsupervised_z_dim_2_50_latent_space_class_distribution.png?raw=Truee" alt="my alt text"/>
  <figcaption>This is my caption text.</figcaption>
</figure>


### Supervised Autoencoder
<figure align="center">
  <img src="https://github.com/Explainable-Artificial-Intelligence/AdversarialAutoencoder/blob/master/doc/results/mnist/mnist_supervised_500_latent_space_class_distribution.png?raw=Truee" alt="my alt text"/>
  <figcaption>This is my caption text.</figcaption>
</figure>

### Semi-supervised Autoencoder
<figure align="center">
  <img src="https://github.com/Explainable-Artificial-Intelligence/AdversarialAutoencoder/blob/master/doc/results/mnist/mnist_semi_supervised_z_dim_2_100_real_dist_and_encoder_dist.png?raw=Truee" alt="my alt text"/>
  <figcaption>This is my caption text.</figcaption>
</figure>

<figure align="center">
  <img src="https://github.com/Explainable-Artificial-Intelligence/AdversarialAutoencoder/blob/master/doc/results/mnist/mnist_semi_supervised_100_gen_images.png?raw=Truee" alt="my alt text"/>
  <figcaption>This is my caption text.</figcaption>
</figure>

## SVHN:

### Unsupervised Autoencoder
<figure align="center">
  <img src="https://github.com/Explainable-Artificial-Intelligence/AdversarialAutoencoder/blob/master/doc/results/svhn/svhn_unsupervised_200_reconstruction_grid.png?raw=Truee" alt="my alt text"/>
  <figcaption>This is my caption text.</figcaption>
</figure>

<figure align="center">
  <img src="https://github.com/Explainable-Artificial-Intelligence/AdversarialAutoencoder/blob/master/doc/results/svhn/svhn_unsupervised_200_latent_space_class_distribution.png?raw=Truee" alt="my alt text"/>
  <figcaption>This is my caption text.</figcaption>
</figure>

### Supervised Autoencoder
<figure align="center">
  <img src="https://github.com/Explainable-Artificial-Intelligence/AdversarialAutoencoder/blob/master/doc/results/svhn/svhn_supervised_250_latent_space_class_distribution.png?raw=Truee" alt="my alt text"/>
  <figcaption>This is my caption text.</figcaption>
</figure>

### Semi-supervised Autoencoder
<figure align="center">
  <img src="https://github.com/Explainable-Artificial-Intelligence/AdversarialAutoencoder/blob/master/doc/results/svhn/schn_semi_supervised_250_latent_space_class_distribution.png?raw=Truee" alt="my alt text"/>
  <figcaption>This is my caption text.</figcaption>
</figure>

<figure align="center">
  <img src="https://github.com/Explainable-Artificial-Intelligence/AdversarialAutoencoder/blob/master/doc/results/svhn/svhn_semi_supervised_500_gen_images.png?raw=Truee" alt="my alt text"/>
  <figcaption>This is my caption text.</figcaption>
</figure>


## Mass Spectrometry data:

### Unsupervised Autoencoder

#### Undercomplete Autoencoder

<figure align="center">
  <img src="https://github.com/Explainable-Artificial-Intelligence/AdversarialAutoencoder/blob/master/doc/results/mass_spec/spectra_generation/undercomplete_autoencoder/1000_mass_specs_spectra_small.png?raw=Truee" alt="my alt text"/>
  <figcaption>This is my caption text.</figcaption>
</figure>

<figure align="center">
  <img src="https://github.com/Explainable-Artificial-Intelligence/AdversarialAutoencoder/blob/master/doc/results/mass_spec/spectra_generation/undercomplete_autoencoder/generated_spectra.png?raw=Truee" alt="my alt text"/>
  <figcaption>This is my caption text.</figcaption>
</figure>

<figure align="center">
  <img src="https://github.com/Explainable-Artificial-Intelligence/AdversarialAutoencoder/blob/master/doc/results/mass_spec/spectra_generation/undercomplete_autoencoder/mz_intensity_distributions.png?raw=Truee" alt="my alt text"/>
  <figcaption>This is my caption text.</figcaption>
</figure>

#### Overcomplete Autoencoder

<figure align="center">
  <img src="https://github.com/Explainable-Artificial-Intelligence/AdversarialAutoencoder/blob/master/doc/results/mass_spec/spectra_generation/overcomplete_autoencoder/1000_mass_specs_spectra_small.png?raw=Truee" alt="my alt text"/>
  <figcaption>This is my caption text.</figcaption>
</figure>

<figure align="center">
  <img src="https://github.com/Explainable-Artificial-Intelligence/AdversarialAutoencoder/blob/master/doc/results/mass_spec/spectra_generation/overcomplete_autoencoder/generated_spectra.png?raw=Truee" alt="my alt text"/>
  <figcaption>This is my caption text.</figcaption>
</figure>

<figure align="center">
  <img src="https://github.com/Explainable-Artificial-Intelligence/AdversarialAutoencoder/blob/master/doc/results/mass_spec/spectra_generation/overcomplete_autoencoder/mz_intensity_distributions_large.png?raw=Truee" alt="my alt text"/>
  <figcaption>This is my caption text.</figcaption>
</figure>

