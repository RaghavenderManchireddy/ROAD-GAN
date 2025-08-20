import tensorflow as tf
from keras import layers, Model

class ResidualBlock(layers.Layer):
    def __init__(self, filters, use_bias=False):
        super(ResidualBlock, self).__init__()
        self.filters = filters
        self.conv1 = layers.Conv2D(filters, 3, padding='same', use_bias=use_bias)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters, 3, padding='same', use_bias=use_bias)
        self.bn2 = layers.BatchNormalization()
        
        # Add a projection shortcut if input channels != filters
        self.projection = None
        
    def build(self, input_shape):
        if input_shape[-1] != self.filters:
            self.projection = layers.Conv2D(self.filters, 1, padding='same', use_bias=False)
        super().build(input_shape)
        
    def call(self, x):
        residual = x if self.projection is None else self.projection(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += residual
        return tf.nn.relu(x)

def build_generator():
    """
    Build and return the generator model.
    Uses U-Net architecture with residual blocks.
    """
    inputs = layers.Input(shape=[128, 128, 3])  # Reduced input size
    
    # Encoder
    encoder_outputs = []
    x = inputs
    
    # Initial convolution
    x = layers.Conv2D(32, 3, padding='same', use_bias=False)(x)  # Reduced initial filters
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    encoder_outputs.append(x)
    
    # Encoder blocks with residual connections
    encoder_filters = [64, 128, 256]  # Reduced number of filters and layers
    for filters in encoder_filters:
        x = layers.Conv2D(filters, 4, strides=2, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        x = ResidualBlock(filters)(x)
        encoder_outputs.append(x)
    
    # Middle blocks
    x = ResidualBlock(256)(x)
    x = ResidualBlock(256)(x)
    
    # Decoder blocks with skip connections and residual blocks
    decoder_filters = [128, 64, 32]  # Match number of encoder layers
    for i, filters in enumerate(decoder_filters):
        x = layers.UpSampling2D(2)(x)
        x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        if i < 2:
            x = layers.Dropout(0.5)(x)
        # Get corresponding encoder output for skip connection
        skip = encoder_outputs[-(i+2)]
        if skip.shape[-1] != filters:
            skip = layers.Conv2D(filters, 1, padding='same')(skip)
        x = layers.concatenate([x, skip])
        x = ResidualBlock(filters)(x)
        x = layers.ReLU()(x)
    
    # Output layer
    x = layers.Conv2D(3, 3, padding='same', activation='tanh')(x)
    
    return Model(inputs=inputs, outputs=x)

def build_discriminator():
    """
    Build and return the discriminator model.
    Uses PatchGAN with reduced size.
    """
    # Input noisy and either real or generated clear image
    noisy_input = layers.Input(shape=[128, 128, 3], name='noisy_input')
    target_input = layers.Input(shape=[128, 128, 3], name='target_input')
    
    x = layers.concatenate([noisy_input, target_input])
    
    # PatchGAN architecture
    filter_sizes = [32, 64, 128]  # Reduced number of filters and layers
    
    for i, filters in enumerate(filter_sizes):
        if i == 0:
            x = layers.Conv2D(filters, 4, strides=2, padding='same')(x)
        else:
            x = layers.Conv2D(filters, 4, strides=2, padding='same', use_bias=False)(x)
            x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
    
    # Output layer for PatchGAN
    x = layers.Conv2D(1, 4, padding='same')(x)
    
    return Model(inputs=[noisy_input, target_input], outputs=x)

if __name__ == "__main__":
    try:
        # CPU optimizations
        tf.config.threading.set_inter_op_parallelism_threads(4)
        tf.config.threading.set_intra_op_parallelism_threads(4)
        
        generator = build_generator()
        discriminator = build_discriminator()
        
        # Print model summaries with memory estimates
        print("\nGenerator Summary:")
        generator.summary()
        total_params_gen = sum(p.numpy().size for p in generator.trainable_variables)
        print(f"Total trainable parameters: {total_params_gen:,}")
        
        print("\nDiscriminator Summary:")
        discriminator.summary()
        total_params_disc = sum(p.numpy().size for p in discriminator.trainable_variables)
        print(f"Total trainable parameters: {total_params_disc:,}")
        
        print(f"\nTotal model parameters: {total_params_gen + total_params_disc:,}")
        
    except Exception as e:
        print(f"Error during model initialization: {e}")
