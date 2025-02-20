import cv2
import matplotlib.pyplot as plt

def show_image_pair(img1, img2):
    fig, axs = plt.subplots(1, 2, figsize=(10, 8))

    image1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    image2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    axs[0].imshow(image1_rgb)
    axs[0].axis('off') 
    axs[0].set_title('Image 1')
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    axs[1].imshow(image2_rgb)
    axs[1].axis('off')
    axs[1].set_title('Image 2')
    axs[1].set_xticks([])
    axs[1].set_yticks([])

    fig.show()


def show_image(img1):

    fig, ax = plt.subplots()

    fig.set_size_inches((10, 8))
    
    image1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(image1_rgb)