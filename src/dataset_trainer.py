# imports
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_dict = pickle.load(open('/home/keys/me_meow/code/python_projects/sign_language_analyzer/lib/data/data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print(score)

f = open('/home/keys/me_meow/code/python_projects/sign_language_analyzer/lib/models/model.p', 'wb')
pickle.dump({'model': model, 'labels': labels}, f)
f.close()