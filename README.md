
# Abhishek-N

# 📝 Task-01: Text Generation with GPT-2 (Fine-Tuning on Custom Dataset)

🚀 **Task-01 of AI/ML Internship at Prodigy Infotech**

In this task, I fine-tuned a **GPT-2 model** on a **custom dataset** to generate **coherent and contextually relevant text**. GPT-2 is a transformer-based language model developed by **OpenAI**.

---

## ✨ Task Overview:
- Fine-tuned GPT-2 using Hugging Face Transformers.
- Custom Dataset used for training to mimic specific writing styles.
- Generated text outputs based on a given prompt.

---

## 🛠 Tech Stack:
- Python
- Hugging Face Transformers
- Google Colab (or Jupyter Notebook)
- PyTorch/TensorFlow backend
- GPT-2 (Pre-trained)

---

## 📂 Project Structure:


---

## ⚙️ How to Run
1. Open the notebook: `gpt2_text_generation.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Upload your own custom dataset in `dataset/custom_dataset.txt`.
3. Run all cells to fine-tune the model.
4. Generate text using a prompt.
5. Output will be saved in `outputs/generated_text_sample.txt`.

---

## 📊 Sample Output:

# 🧠 Task-02: Image Generation with Pre-trained Models

### 📍 Internship at Prodigy Infotech  
**Domain:** AI / Machine Learning  
**Task:** Generate images from text prompts using pre-trained generative models.

---

## 🚀 Objective

To explore and implement **text-to-image generation** using cutting-edge **pre-trained models** like **Stable Diffusion** and **DALL·E-mini**, transforming natural language descriptions into visual outputs.

---

## 🧰 Tools & Technologies

- 🐍 Python
- 🤗 Hugging Face `diffusers`
- 🔥 PyTorch
- 🧠 Stable Diffusion v1.4 (`CompVis`)
- 🖼️ DALL·E-mini / Craiyon (optional)

---

## 🧪 Experiment Setup

- Used Hugging Face’s `diffusers` to access `Stable Diffusion v1-4`.
- Loaded model to `cuda` or `cpu` based on availability.
- Prompt used for generation:
  

- Generated and saved image output as `futuristic_city.png`.

---

## 📸 Sample Output

(./futuristic_city.png)
C:\Users\a<img width="512" height="512" alt="futuristic_city (2)" src="https://github.com/user-attachments/assets/ecf2c2cc-a0f1-44db-9909-e51f2a8829f6" />
bhi\Downloads\futuristic_city (2).png

> *Image generated using Stable Diffusion based on the text prompt above.*

---

## 📌 Key Learnings

- Understanding latent diffusion techniques for image generation.
- Prompt engineering – how prompt detail impacts image fidelity.
- Comparing generation quality between DALL·E-mini and Stable Diffusion.

---

## 💡 How to Run the Code

### 🔗 [Google Colab Notebook](https://colab.research.google.com/drive/1f4X8hrYkroKXWjO7zWqbywSxPujz6eh1?usp=sharing)

```bash
# 1. Clone this repository or open the notebook
# 2. Run the cells step by step
# 3. Provide your Hugging Face token when prompted
# 4. Customize your own text prompts

# 📜 Task-03: Text Generation using Markov Chains

🚀 **Task-03 of AI/ML Internship at Prodigy Infotech**

In this task, I built a **Text Generation model using Markov Chains**, a statistical NLP approach that predicts the next word based on the current state (n-grams).

---

## 🛠️ What I Did:
- Built a word-based **Markov Chain model** from scratch in Python.
- Generated coherent text that mimics the style of the input dataset.
- Learned core concepts of **early probabilistic NLP models**.

---

## 📂 Project Structure:


---

## ⚙️ How to Run
```bash
python markov_chain_text_generation.py
output:
Once upon a time, there was a small village where the people loved stories. Every evening, they would gather around the fire...

---

## 🐍 **Python Code (markov_chain_text_generation.py)**

```python
import random
import os

# Load dataset
with open('dataset/sample_input_text.txt', 'r') as file:
    data = file.read()

# Tokenize words
words = data.split()

# Build Markov Chain dictionary (n=2 for bi-gram model)
markov_chain = {}
for i in range(len(words)-2):
    key = (words[i], words[i+1])
    next_word = words[i+2]
    if key not in markov_chain:
        markov_chain[key] = []
    markov_chain[key].append(next_word)

# Text Generation
def generate_text(chain, length=100):
    seed = random.choice(list(chain.keys()))
    output = list(seed)

    for _ in range(length):
        current_key = tuple(output[-2:])
        next_words = chain.get(current_key, None)
        if not next_words:
            break
        next_word = random.choice(next_words)
        output.append(next_word)

    return ' '.join(output)

# Generate sample text
generated_text = generate_text(markov_chain, 100)

# Save output
os.makedirs('outputs', exist_ok=True)
with open('outputs/generated_text_sample.txt', 'w') as out_file:
    out_file.write(generated_text)

print("Generated Text:\n")
print(generated_text)
cd MarkovChain-Text-Generation
git init
git add .
git commit -m "Task-03 Markov Chain Text Generation Completed"
git remote add origin https://github.com/YourUsername/MarkovChain-Text-Generation.git
git push -u origin master

# 🖼️ Task-04: Image-to-Image Translation with Pix2Pix (cGAN)

🚀 **Task-04 of AI/ML Internship at Prodigy Infotech**

In this task, I implemented **Image-to-Image Translation** using **Conditional Generative Adversarial Networks (cGANs)** through the **Pix2Pix model**. The goal was to learn how sketches or edges can be transformed into realistic images using deep learning.

---

## 🧠 Key Concepts Explored:
- Understanding **cGAN architecture** (Generator + Discriminator).
- Translating **sketches to realistic images**.
- Training and testing with paired datasets.
- Utilizing **pre-trained Pix2Pix models** for edge ➝ shoe & sketch ➝ photo transformations.

---

## 📂 Project Structure:

---

## 🖼️ Example Output:
| Input Image | Translated Output |
|-------------|-------------------|
| ![Input](images/input_edge_image.jpg) | ![Output](images/output_translated_image.jpg) |

---

## ⚙️ How to Run:
1. Open `pix2pix_image_translation.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Use pre-trained models from TensorFlow Hub or custom datasets.
3. Run inference on edge/sketch images to get the photo-realistic output.
4. Save generated outputs into `/images/`.

---

## 📚 Libraries Used:
- TensorFlow / Keras
- TensorFlow Datasets (for paired data)
- Pix2Pix pre-trained models from TensorFlow Hub

---

## 📝 Sample Code Snippet:
```python
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import matplotlib.pyplot as plt

# Load pre-trained Pix2Pix model
model = hub.load('https://tfhub.dev/google/pix2pix/edges2shoes/1')

# Load test image (Edge Image)
input_image = tf.io.read_file('images/input_edge_image.jpg')
input_image = tf.image.decode_jpeg(input_image)
input_image = tf.image.convert_image_dtype(input_image, tf.float32)
input_image = tf.image.resize(input_image, (256, 256))
input_image = tf.expand_dims(input_image, 0)

# Generate output
output_image = model(input_image)

# Save and Display Output
output = tf.squeeze(output_image)
plt.imshow(output)
plt.axis('off')
plt.savefig('images/output_translated_image.jpg')
plt.show()


---
output:
<img width="512" height="512" alt="image" src="https://github.com/user-attachments/assets/50b755ba-a05f-4407-8028-4cd0587fa0fb" />
![Uploading _7000051.jpg…]()


## 🖥️ **GitHub Push Commands**
```bash
cd Pix2Pix-Image-Translation
git init
git add .
git commit -m "Task-04 Pix2Pix Image-to-Image Translation Completed"
git remote add origin https://github.com/YourUsername/Pix2Pix-Image-Translation.git
git push -u origin master

# 🎨 Task-05: Neural Style Transfer (NST) using TensorFlow

🚀 **Task-05 of AI/ML Internship at Prodigy Infotech**

In this task, I implemented **Neural Style Transfer (NST)** — a fascinating deep learning technique that blends the **content of one image** with the **artistic style of another** using **VGG19 (pre-trained CNN model)**.

---

## 🧠 What I Learned:
- Feature extraction using **Convolutional Neural Networks (CNNs)**
- Calculating **Gram matrices** for style representation.
- Image optimization to match both **content** and **style features**.
- Hands-on practice with **TensorFlow/Keras** for image aesthetics in AI.

---

## 🖼️ Example Images:
| Content Image | Style Image | Stylized Output |
|---------------|-------------|-----------------|
| ![Content](images/content_image.jpg) | ![Style](images/style_image.jpg) | ![Output](images/stylized_output_image.jpg) |

---

## ⚙️ How to Run:
1. Open `neural_style_transfer.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Upload your own **content** and **style** images.
3. Run the notebook to perform style transfer.
4. View and save the generated stylized output in the `images/` folder.

---

## 📝 Libraries Used:
- TensorFlow / Keras
- Matplotlib
- NumPy
- PIL (Python Imaging Library)

---

## Sample Code Snippet:
```python
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications import vgg19
from PIL import Image
import numpy as np

# Load content and style images
content_image = Image.open('images/content_image.jpg').resize((512, 512))
style_image = Image.open('images/style_image.jpg').resize((512, 512))

# Preprocess images
def preprocess(img):
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

content = preprocess(content_image)
style = preprocess(style_image)

# Load VGG19 model for feature extraction
vgg = vgg19.VGG19(include_top=False, weights='imagenet')
vgg.trainable = False

# Extract feature maps...
# (Neural Style Transfer logic continues here...)

# Save the output image
stylized_image = Image.fromarray((generated_image[0] * 255).astype('uint8'))
stylized_image.save('images/stylized_output_image.jpg')


---

## 🖥️ GitHub Push Commands:
```bash
cd Neural-Style-Transfer-TensorFlow
git init
git add .
git commit -m "Task-05 Neural Style Transfer using TensorFlow Completed"
git remote add origin https://github.com/YourUsername/Neural-Style-Transfer-TensorFlow.git
git push -u origin master


out put:
Stylized Output Image:
C:\Users\abhi\OneDrive\Desktop\New folder\image.png
