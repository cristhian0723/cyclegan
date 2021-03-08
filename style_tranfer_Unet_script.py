# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tkinter import filedialog
import os

# image proccesin 
from skimage.transform import resize
from scipy import misc
import seaborn as sns

# tensorflow
import tensorflow as tf
from tensorflow.keras import Input,Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add, Conv2DTranspose, UpSampling2D, MaxPooling2D,     ZeroPadding2D, LeakyReLU, Concatenate
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# %%
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth (physical_devices[0],True)


# %%
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)




#path_style = filedialog.askdirectory() # style-------------------------


# %%
#path_images_to_transform = filedialog.askdirectory() # to transfer -------------------------


# %%
# save images
path_save = filedialog.askdirectory() # save the images ----------------------

# %%
path_save_weights = filedialog.askdirectory() # save the weights------------


# %%
batch_size = 1
input_shape=(400,640,3)


# %%
def build_discriminator(input_shape):

    def conv4(layer_input,filters, stride = 2, norm=True):
        y = Conv2D(filters, kernel_size=4, strides=stride
               , padding='same')(layer_input)

        if norm:
            y = tfa.layers.InstanceNormalization(axis = -1)(y)

        y = LeakyReLU(0.2)(y)

        return y
    
    #input_shape = (400, 400, 3)
    input_layer = Input(shape=input_shape)
    
    disc_n_filters = 32

    y = conv4(input_layer, disc_n_filters, stride = 2, norm = False) 
    y = conv4(y, disc_n_filters*2, stride = 2)
    y = conv4(y, disc_n_filters*4, stride = 2)
    y = conv4(y, disc_n_filters*4, stride = 2)
    y = conv4(y, disc_n_filters*8, stride = 1)

    output = Conv2D(1, kernel_size=4, strides=1, padding='same',activation="sigmoid")(y) 

    return Model(input_layer, output)


# %%
build_discriminator(input_shape = input_shape).summary()


# %%
def build_generator(input_shape):
    """Unet"""

    def downsample(layer_input, filters, f_size=4, s_size = 2):
        d = Conv2D(filters, kernel_size=f_size
            , strides=s_size, padding='same')(layer_input)
        d = tfa.layers.InstanceNormalization(axis = -1)(d)
        d = Activation('relu')(d)

        return d

    def upsample(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same')(u)
        u = tfa.layers.InstanceNormalization(axis = -1)(u)
        u = Activation('relu')(u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)

        u = Concatenate()([u, skip_input])
        return u

    # Image input
    #input_shape = input_shape
    input_layer = Input(shape=input_shape)

    # Downsampling 1
    gen_n_filters = 32 # alter
    
    #d11 = downsample(input_layer, gen_n_filters,s_size=1)
    #d22 = downsample(d11, gen_n_filters,s_size=1)
    #d33 = downsample(d22, gen_n_filters,s_size=1)
    d1 = downsample(input_layer, gen_n_filters)
    d2 = downsample(d1, gen_n_filters*2)
    d3 = downsample(d2, gen_n_filters*4)
    d4 = downsample(d3, gen_n_filters*8)

    # Upsampling 2
    u1 = upsample(d4, d3, gen_n_filters*4)
    u2 = upsample(u1, d2, gen_n_filters*2)
    u3 = upsample(u2, d1, gen_n_filters)

    u4 = UpSampling2D(size=2)(u3)

    output = Conv2D(3, kernel_size=4, strides=1
           , padding='same', activation='tanh')(u4)

    return Model(input_layer, output)


# %%
build_generator(input_shape = input_shape).summary()


# %%
common_optimizer = Adam(0.0002,0.5)


# %%
# Build and compile generator networks
discriminatorA = build_discriminator(input_shape = input_shape)
discriminatorB = build_discriminator(input_shape = input_shape)


# %%
discriminatorA.compile(loss = "mse", optimizer = common_optimizer, metrics = ["acc"])
discriminatorB.compile(loss = "mse", optimizer = common_optimizer, metrics = ['acc'])


# %%
# Build generator networks
generatorAToB = build_generator(input_shape = input_shape)
generatorBToA = build_generator(input_shape = input_shape)


# %%
discriminatorA.trainable = False
discriminatorB.trainable = False


# %%
# Create and adversarial network3
inputA = Input(shape=input_shape)
inputB = Input(shape=input_shape)


# %%
# Generator networks to generate fake images
generatedA = generatorBToA(inputB) # fake B
generatedB = generatorAToB(inputA) # fake A


# %%
# Use the discriminator networks to predict whether each generated image is real or fake
valid_A = discriminatorA(generatedA)
valid_B = discriminatorB(generatedB)


# %%
# Reconstruct the original images using the generator networks again
reconstructedA = generatorBToA(generatedB)
reconstructedB = generatorAToB(generatedA)


# %%
# Use the generator networks to generate fake images
generatedAId = generatorBToA(inputA)
generatedBId = generatorAToB(inputB)


# %%
# Create a Keras model and specify the inputs and outputs for the network
adversarial_model = Model(inputs=[inputA, inputB],
                          outputs=[valid_A, valid_B, reconstructedA, reconstructedB, generatedAId, generatedBId])


# %%
adversarial_model.compile(loss=['mse', 'mse', 'mae', 'mae', 'mae', 'mae'],
                          loss_weights=[1, 1, 10, 10, 2, 2],
                          optimizer=common_optimizer)


# %%
valid = np.ones((batch_size, 25, 40, 1))
fake = np.zeros((batch_size, 25, 40, 1))


# %%
train_datagen_parameters = ImageDataGenerator(
    rescale=1/255,
    #samplewise_center = True,
    horizontal_flip=True,
    width_shift_range = 20,
    height_shift_range = 20,
    fill_mode="reflect",
    #rotation_range=20,
    data_format="channels_last")

# %%
train_generator_styleA = train_datagen_parameters.flow_from_directory(
    directory='/media/cristhian/PERFORMANCE/proyectos de deep learning/Data',
    class_mode=None,
    classes=["winter images"],
    target_size=(input_shape[0],input_shape[1]),
    batch_size=batch_size,
    shuffle=False, # the same data 
    seed=10
)

# %% 
train_generator_styleB = train_datagen_parameters.flow_from_directory(
    directory='/media/cristhian/PERFORMANCE/proyectos de deep learning/Data',
    class_mode=None,
    classes=["medellin images"],
    target_size=(input_shape[0],input_shape[1]),
    batch_size=batch_size,
    shuffle=False,
    seed=10
)
# %%
path = '/media/cristhian/PERFORMANCE/proyectos de deep learning/Data'
styleA = os.path.join(path,"winter images")
styleB = os.path.join(path,"medellin images")

num_images_styleA = len(os.listdir(styleA))
num_images_styleB = len(os.listdir(styleB))

print("Num of images {}".format(num_images_styleB))
print("index",int(num_images_styleB/batch_size))

# %%
def dir_weights(style_path_result,model_name):
    path_weights = "/media/cristhian/PERFORMANCE/proyectos de deep learning/style transfer code folder/save_results/{}/model weights 2/{}/".format(style_path_result,model_name)
    checkpoint_dir = os.path.dirname(path_weights)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    epoch_load = latest[-7:-5]
    return latest,int(epoch_load)

# %%
def trainin(epochs,batch_size,discriminatorA,discriminatorB,generatorAToB,generatorBToA,save_model = False,path_save = path_save,path_save_weights = path_save_weights,load_weights_model = False): 

    dis_losses = []
    gen_losses = []
    if load_weights_model == True:
        discriminatorA.load_weights(dir_weights("winter images","discriminatorA")[0])
        discriminatorB.load_weights(dir_weights("winter images","discriminatorB")[0])
        adversarial_model.load_weights(dir_weights("winter images","adversarial_model")[0])
        epoch_load = dir_weights("winter images","adversarial_model")[1]
    else:
        pass

    # Training loop 
    for epoch in range(1,epochs):
        
        if load_weights_model == True:
            epoch = epoch + epoch_load
        else:
            pass

        for index in range(1,int(num_images_styleB/batch_size)+1):
            
            #Sample images
            batchA = next(train_generator_styleA)
            batchB = next(train_generator_styleB)

            #if (batchA.shape[0] != batch_size) or (batchB.shape[0] != batch_size):
            #    print(batchA.shape[0] != batch_size, batchB.shape[0] != batch_size)
            #    print("Error in batch shape at epoch {}, index {}".format(epoch,index))
            #    continue
            #else:
            #    pass

            # rescale
            batchA = (batchA - 0.5)/0.5
            batchB = (batchB - 0.5)/0.5
            
            #Traslate images to opposite domain
            generatedB = generatorAToB.predict(batchA)
            generatedA = generatorBToA.predict(batchB)
            
            # Train the discriminator A on real and fake images
            dALoss1 = discriminatorA.train_on_batch(batchA, valid)
            dALoss2 = discriminatorA.train_on_batch(generatedA, fake)
            
            # Train the discriminator B on real and fake images
            dBLoss1 = discriminatorB.train_on_batch(batchB, valid)
            dBLoss2 = discriminatorB.train_on_batch(generatedB, fake)
            
            # Calculate the total discriminator loss
            d_loss = 0.5 * np.add(0.5 * np.add(dALoss1, dALoss2), 0.5 * np.add(dBLoss1, dBLoss2))
            
            # Train the generator networks
            g_loss = adversarial_model.train_on_batch([batchA, batchB],
                                                    [valid, valid, batchA, batchB, batchA, batchB])
            
            
            dis_losses.append(d_loss)
            gen_losses.append(g_loss)

            #(num_images_styleB - 1)

            if (index % 100 == 0) and (index != 0):
                print("epoch {}, bath_size {}, Total {:.6f}, discriminatorA {:.6f}, discriminatorB {:.6f}, reconstructedA {:.6f}, reconstructedB {:.6f}, generatorBToA {:.6f}, generatorAToB {:.6f}".format(epoch,index,np.array(gen_losses)[-1,0],np.array(gen_losses)[-1,1],np.array(gen_losses)[-1,2],np.array(gen_losses)[-1,3],np.array(gen_losses)[-1,4],
                            np.array(gen_losses)[-1,5],np.array(gen_losses)[-1,6]))
            
             # endo of loop
            
            # Sample and save images after every 10 epochs
            if index % 100 == 0:
                #print("Epoch:{}".format(epoch))
                
                #random_image = np.random.randint(0,len(trainB))
                
                generatedB_test = generatorAToB.predict(batchA)
                generatedA_test = generatorBToA.predict(batchB)

                # Get reconstructed images
                reconsA_test = generatorBToA.predict(generatedB_test)
                reconsB_test = generatorAToB.predict(generatedA_test)
                

                # Save original, generated and reconstructed images
                fig, axs  = plt.subplots(2,3,figsize = (10,5))
                axs[0,0].set_title("From A to B")
                axs[0,0].imshow(batchA[0] * 0.5 + 0.5)
                axs[0,1].imshow(generatedB_test[0].astype("float32") * 0.5 +0.5)
                axs[0,2].imshow(reconsA_test[0].astype("float32") * 0.5 + 0.5)


                axs[1,0].set_title("From B to A")      
                axs[1,0].imshow(batchB[0] * 0.5 + 0.5)
                axs[1,1].imshow(generatedA_test[0].astype("float32") * 0.5 + 0.5)
                axs[1,2].imshow(reconsB_test[0].astype("float32") * 0.5 +0.5)
                plt.savefig(path_save + "/results_{}".format(epoch))
                plt.pause(0.05)
                plt.close()
        
        
        # Save the weights--------------------------------------------------    
        if (save_model == True) and (epoch % 5 == 0): # every 10 epochs
            discriminatorA.save_weights(path_save_weights + "/discriminatorA" + "/discriminatorA-{}.ckpt".format(epoch))
            discriminatorB.save_weights(path_save_weights + "/discriminatorB" + "/discriminatorB-{}.ckpt".format(epoch))
            generatorAToB.save_weights(path_save_weights + "/generatorAToB" + "/generatorAToB-{}.ckpt".format(epoch))
            generatorBToA.save_weights(path_save_weights + "/generatorBToA" + "/generatorBToA-{}.ckpt".format(epoch))
            adversarial_model.save_weights(path_save_weights + "/adversarial_model" + "/adversarial_model-{}.ckpt".format(epoch))
            print("model saved at epoch: {}".format(epoch))
        else:
            pass  

    return np.array(dis_losses), np.array(gen_losses )



# %%
dis_losses, gen_losses  = trainin(
    epochs=300,batch_size=batch_size,
    discriminatorA=discriminatorA,discriminatorB=discriminatorB,generatorAToB=generatorAToB,
    generatorBToA=generatorBToA,save_model=True,load_weights_model=False)
np.save(path_save_weights + "/dis_losses",dis_losses)
np.save(path_save_weights + "/gen_losses",gen_losses)


# %%
def plot_results(from_to,path_save,name,display_num = 10, save = False):
    for i in range(display_num):

        batchB = next(train_generator_styleB)

        # rescale
        batchB = (batchB - 0.5)/0.5
        
        plt.figure(figsize=(10,10))
        predict = from_to.predict(batchB[0].reshape((1,400,400,3)))
        predict = predict.reshape((400,400,3)).astype("float32")
        plt.imshow(predict * 0.5 +0.5)
        plt.axis(False)
        if save == True:
            plt.savefig(path_save + "/{}_{}".format(name,i))
            print("Image saved")
        else:
            continue
        plt.pause(2)
        plt.close()
    return print("ok")

# %%
plot_results(from_to=generatorBToA,path_save=path_save,name="mari",save=False,display_num=50)
