import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt

def load_and_process_image(image_path):
    img = tf.image.resize(tf.image.decode_image(tf.io.read_file(image_path), channels=3), [512, 512])
    return img[tf.newaxis, ...] / 255.0

model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

content_image = load_and_process_image('/content/abcd.png')
style_image = load_and_process_image('/content/abcd.png')

stylized_image = model(content_image, style_image)[0]

plt.figure(figsize=(12, 4))
for i, img in enumerate([content_image, style_image, stylized_image], 1):
    plt.subplot(1, 3, i)
    plt.imshow(img[0])
    plt.axis('off')
plt.show()
