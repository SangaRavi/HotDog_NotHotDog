from keras.models import model_from_json
from pre_process_dataset import *
import matplotlib.pyplot as plt
from keras.models import load_model

test_images = data_test_data_with_label()

model = load_model('Model/model.h5')
print("Loaded model from disk")


fig = plt.figure(figsize=(14, 14))

for cnt, data in enumerate(test_images[:]):
    y= fig.add_subplot(6, 5, cnt+1)
    img = data[0]
    data = img.reshape(1, 64, 64, 3)
    model_out = model.predict([data])

    if np.argmax(model_out) == 1:
        str_label = 'Not HotDog'
    else:
        str_label = 'HotDog'

    b, g, r = cv2.split(img)  # get b,g,r
    img = cv2.merge([r, g, b])  # switch it to rgb
    y.imshow(img)
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)

plt.show()