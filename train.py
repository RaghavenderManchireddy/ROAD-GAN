import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from model import build_generator, build_discriminator
from keras.optimizers import Adam
from keras.applications import VGG19
from keras.applications.vgg19 import preprocess_input

# Data preparation
def load_data(noisy_dir, clear_dir, img_size=128):  # Reduced image size
    noisy_images = []
    clear_images = []
    
    for noisy_file in os.listdir(noisy_dir):
        clear_path = os.path.join(clear_dir, noisy_file)
        if os.path.exists(clear_path):
            # Load and process noisy image (3 channels)
            noisy = cv2.imread(os.path.join(noisy_dir, noisy_file))
            noisy = cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB)
            noisy = cv2.resize(noisy, (img_size, img_size))
            noisy = (noisy / 127.5) - 1.0  # Normalize to [-1, 1]
            
            # Load and process clear image (3 channels)
            clear = cv2.imread(clear_path)
            clear = cv2.cvtColor(clear, cv2.COLOR_BGR2RGB)
            clear = cv2.resize(clear, (img_size, img_size))
            clear = (clear / 127.5) - 1.0  # Normalize to [-1, 1]
            
            noisy_images.append(noisy)
            clear_images.append(clear)
    
    return np.array(noisy_images, dtype=np.float32), np.array(clear_images, dtype=np.float32)

# Initialize VGG model for perceptual loss
vgg = VGG19(include_top=False, weights='imagenet')
vgg.trainable = False
# Use fewer layers for perceptual loss to reduce computation
content_layers = ['block1_conv2', 'block2_conv2']  # Reduced number of layers
content_outputs = [vgg.get_layer(name).output for name in content_layers]
content_model = tf.keras.Model(vgg.input, content_outputs)

# Model configuration
generator = build_generator()
discriminator = build_discriminator()

generator_optimizer = Adam(2e-4, beta_1=0.5)  # Slightly increased learning rate
discriminator_optimizer = Adam(2e-4, beta_1=0.5)

# Loss functions
bce = tf.losses.BinaryCrossentropy(from_logits=True)
mae = tf.losses.MeanAbsoluteError()

def perceptual_loss(real_image, generated_image):
    # Convert from [-1, 1] to [0, 255]
    real_image = tf.cast((real_image + 1) * 127.5, tf.float32)
    generated_image = tf.cast((generated_image + 1) * 127.5, tf.float32)
    
    # Preprocess for VGG
    real_features = content_model(preprocess_input(real_image))
    generated_features = content_model(preprocess_input(generated_image))
    
    # Calculate loss for each layer
    loss = 0
    for real_feat, gen_feat in zip(real_features, generated_features):
        loss += tf.reduce_mean(tf.abs(real_feat - gen_feat))
    
    return loss

def gradient_penalty(discriminator, real_samples, fake_samples, noisy_samples):
    batch_size = tf.shape(real_samples)[0]
    alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0, dtype=tf.float32)
    diff = tf.cast(fake_samples, tf.float32) - tf.cast(real_samples, tf.float32)
    interpolated = tf.cast(real_samples, tf.float32) + alpha * diff
    
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = discriminator([noisy_samples, interpolated], training=True)
        
    grads = gp_tape.gradient(pred, interpolated)
    # Calculate gradient norm properly for each sample in batch
    grad_norms = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    gradient_penalty = tf.reduce_mean(tf.square(grad_norms - 1.0))
    return gradient_penalty

def generator_loss(disc_output, gen_output, target):
    gan_loss = bce(tf.ones_like(disc_output), disc_output)
    l1_loss = mae(target, gen_output)
    percep_loss = perceptual_loss(target, gen_output)
    # Adjusted weights
    return gan_loss + (10 * l1_loss) + (0.1 * percep_loss)  # Reduced perceptual loss weight

def discriminator_loss(disc_real_output, disc_gen_output):
    real_loss = bce(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = bce(tf.zeros_like(disc_gen_output), disc_gen_output)
    return real_loss + generated_loss

# Training loop
def train(train_noisy, train_clear, val_noisy, val_clear, epochs=80, batch_size=2):
    # Initialize history dictionary
    history = {
        'train_gen_loss': [],
        'train_disc_loss': [],
        'val_gen_loss': [],
        'val_disc_loss': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    # Checkpoint setup
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)
    
    # Ensure training_progress directory exists
    if not os.path.exists('training_progress'):
        os.makedirs('training_progress')

    # Prepare datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((train_noisy, train_clear)) \
        .shuffle(buffer_size=len(train_noisy)) \
        .batch(batch_size) \
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((val_noisy, val_clear)) \
        .batch(batch_size) \
        .prefetch(buffer_size=tf.data.AUTOTUNE)

    # Get 6 fixed validation samples for progress visualization
    num_fixed_samples = 6
    fixed_val_noisy_samples = []
    fixed_val_clear_samples = []
    try:
        # Take enough batches to get 6 samples, assuming batch_size might be small
        val_samples_collected = 0
        for batch in val_dataset:
            needed = num_fixed_samples - val_samples_collected
            take_from_batch = min(needed, batch[0].shape[0])
            fixed_val_noisy_samples.append(batch[0][:take_from_batch])
            fixed_val_clear_samples.append(batch[1][:take_from_batch])
            val_samples_collected += take_from_batch
            if val_samples_collected >= num_fixed_samples:
                break
        # Concatenate batches if necessary
        fixed_val_noisy_samples = tf.concat(fixed_val_noisy_samples, axis=0)
        fixed_val_clear_samples = tf.concat(fixed_val_clear_samples, axis=0)

        if val_samples_collected < num_fixed_samples:
            print(f"Warning: Only found {val_samples_collected} validation samples. Using these for visualization.")
            num_fixed_samples = val_samples_collected # Adjust number of samples if less than 5 found

    except StopIteration:
        print("Error: Validation dataset is empty. Cannot select fixed samples.")
        return history # Or handle appropriately

    if num_fixed_samples == 0:
         print("Error: No validation samples available for visualization.")
         # Potentially continue training without visualization or return
         # return history

    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        print(f'\nEpoch {epoch+1}/{epochs}')
        epoch_gen_loss = []
        epoch_disc_loss = []
        
        try:
            for noisy_batch, clear_batch in train_dataset:
                with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                    # Generate images
                    gen_output = generator(noisy_batch, training=True)
                    
                    # Discriminator outputs
                    disc_real_output = discriminator([noisy_batch, clear_batch], training=True)
                    disc_gen_output = discriminator([noisy_batch, gen_output], training=True)
                    
                    # Calculate losses
                    gen_loss = generator_loss(disc_gen_output, gen_output, clear_batch)
                    disc_loss = discriminator_loss(disc_real_output, disc_gen_output)
                    
                    # Add gradient penalty
                    gp = gradient_penalty(discriminator, clear_batch, gen_output, noisy_batch)
                    disc_loss += 10 * gp # WGAN-GP term
                    
                # Calculate gradients
                generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
                discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
                
                # Apply gradients
                generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
                discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))
                
                # Store batch losses
                epoch_gen_loss.append(gen_loss.numpy())
                epoch_disc_loss.append(disc_loss.numpy())
                
        except Exception as e:
            print(f"Error during training step: {e}")
            continue # Skip to next epoch if error occurs
            
        # Validation loop
        val_gen_losses = []
        val_disc_losses = []
        all_val_preds = []
        all_val_labels = []
        
        for val_noisy_batch, val_clear_batch in val_dataset:
            val_gen_output = generator(val_noisy_batch, training=False)
            val_disc_real = discriminator([val_noisy_batch, val_clear_batch], training=False)
            val_disc_gen = discriminator([val_noisy_batch, val_gen_output], training=False)
            
            val_gen_loss = generator_loss(val_disc_gen, val_gen_output, val_clear_batch)
            val_disc_loss = discriminator_loss(val_disc_real, val_disc_gen)
            val_gen_losses.append(val_gen_loss.numpy())
            val_disc_losses.append(val_disc_loss.numpy())
            
            # Store discriminator predictions and labels for metrics
            real_labels = np.ones_like(val_disc_real.numpy())
            gen_labels = np.zeros_like(val_disc_gen.numpy())
            all_val_preds.extend(val_disc_real.numpy().flatten())
            all_val_preds.extend(val_disc_gen.numpy().flatten())
            all_val_labels.extend(real_labels.flatten())
            all_val_labels.extend(gen_labels.flatten())
        
        # Convert predictions to binary values
        # Handle potential division by zero if no predictions
        if len(all_val_labels) > 0:
            binary_preds = (np.array(all_val_preds) > 0).astype(int)
            binary_labels = np.array(all_val_labels).astype(int)
            
            # Calculate metrics with zero_division handling
            precision = precision_score(binary_labels, binary_preds, zero_division=0)
            recall = recall_score(binary_labels, binary_preds, zero_division=0)
            f1 = f1_score(binary_labels, binary_preds, zero_division=0)
        else:
            precision, recall, f1 = 0.0, 0.0, 0.0

        # Update history
        history['train_gen_loss'].append(np.mean(epoch_gen_loss))
        history['train_disc_loss'].append(np.mean(epoch_disc_loss))
        history['val_gen_loss'].append(np.mean(val_gen_losses))
        history['val_disc_loss'].append(np.mean(val_disc_losses))
        history['precision'].append(precision)
        history['recall'].append(recall)
        history['f1'].append(f1)
        
        # Print epoch statistics
        print(f'Gen Loss: {np.mean(epoch_gen_loss):.4f}  Disc Loss: {np.mean(epoch_disc_loss):.4f}')
        print(f'Val Gen Loss: {np.mean(val_gen_losses):.4f}  Val Disc Loss: {np.mean(val_disc_losses):.4f}')
        print(f'Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}')
        
        # Save checkpoint and generate sample image every 5 epochs
        if (epoch + 1) % 5 == 0:  # More frequent checkpoints
            checkpoint.save(file_prefix=checkpoint_prefix)
        
        # --- Generate and plot results for the fixed validation samples ---
        if num_fixed_samples > 0: # Only plot if we have samples
            val_gen_samples = generator(fixed_val_noisy_samples, training=False)

            # Create a single figure for all samples
            plt.figure(figsize=(15, 3 * num_fixed_samples)) # Adjust height based on number of samples
            plt.suptitle(f'Epoch {epoch+1} Progress Samples', fontsize=16)

            for i in range(num_fixed_samples):
                # Noisy Input
                plt.subplot(num_fixed_samples, 3, i * 3 + 1)
                plt.imshow((fixed_val_noisy_samples[i].numpy() + 1) / 2)
                plt.title(f'Sample {i+1}: Input Noisy')
                plt.axis('off')

                # Generated Clear
                plt.subplot(num_fixed_samples, 3, i * 3 + 2)
                plt.imshow((val_gen_samples[i].numpy() + 1) / 2)
                plt.title(f'Sample {i+1}: Generated Clear')
                plt.axis('off')

                # Ground Truth Clear
                plt.subplot(num_fixed_samples, 3, i * 3 + 3)
                plt.imshow((fixed_val_clear_samples[i].numpy() + 1) / 2)
                plt.title(f'Sample {i+1}: Ground Truth Clear')
                plt.axis('off')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
            plt.savefig(f'training_progress/epoch_{epoch+1}_samples.png')
            plt.close()
    
    return history

if __name__ == "__main__":
    # Load data
    noisy_images, clear_images = load_data('database/noisy', 'database/clear')
    print(f"Loaded data shapes - noisy: {noisy_images.shape}, clear: {clear_images.shape}")
    
    # Split into train/validation
    split = int(0.8 * len(noisy_images))
    train_noisy, train_clear = noisy_images[:split], clear_images[:split]
    val_noisy, val_clear = noisy_images[split:], clear_images[split:] # Create validation split
    print(f"Training data shapes - noisy: {train_noisy.shape}, clear: {train_clear.shape}")
    print(f"Validation data shapes - noisy: {val_noisy.shape}, clear: {val_clear.shape}")

    # Check if validation data exists
    if val_noisy.shape[0] == 0 or val_clear.shape[0] == 0:
        print("Error: Not enough data for validation split.")
    else:
        # Start training and get history
        history = train(train_noisy, train_clear, val_noisy, val_clear) # Pass validation data to train
        
        # Save final model
        generator.save('weather_gan_generator.h5')
        
        # Plot training history
        plt.figure(figsize=(15, 5))
        
        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(history['train_gen_loss'], label='Generator Loss')
        plt.plot(history['train_disc_loss'], label='Discriminator Loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot validation metrics
        plt.subplot(1, 2, 2)
        plt.plot(history['val_gen_loss'], label='Validation Gen Loss')
        plt.plot(history['val_disc_loss'], label='Validation Disc Loss')
        plt.title('Validation Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.savefig('training_progress/training_history.png')
        plt.close()
        
        # Plot classification metrics
        plt.figure(figsize=(15, 5))
        plt.plot(history['precision'], label='Precision')
        plt.plot(history['recall'], label='Recall')
        plt.plot(history['f1'], label='F1 Score')
        plt.title('Classification Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.savefig('training_progress/classification_metrics.png')
        plt.close()
