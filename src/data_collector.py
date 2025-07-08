# imports
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
import cv2
import yaml

with open('/home/keys/me_meow/code/python_projects/sign_language_analyzer/src/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# put your own directory in the config.yaml file 
dir = config['directory']['dir']
if not os.path.exists(dir):
    os.makedirs(dir)

number_of_classes = config['values']['number_of_classes']
dataset_size = config['values']['dataset_size']

# change values to either 1 or 2 in the config in case of errors
cap = cv2.VideoCapture(config['webcam']['device_index'])

for j in range(number_of_classes):
    if not os.path.exists(os.path.join(dir, str(j))):
        os.makedirs(os.path.join(dir, str(j)))

    print('Collecting data for class {}'.format(j))

    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(dir, str(j), '{}.jpg'.format(counter)), frame)

        counter += 1

cap.release()
cv2.destroyAllWindows()