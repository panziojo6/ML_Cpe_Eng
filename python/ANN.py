import os
import numpy as np
import rasterio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras import models, layers, optimizers
def read_raster(file_path):
    with rasterio.open(file_path) as src:
        return src.read(1)

data_dir_SAVI = '../../data/Index/savi'
data_dir_NDVI = '../../data/Index/ndvi'

ndvi_files = [os.path.join(data_dir_NDVI, f) for f in os.listdir(data_dir_NDVI) if f.endswith('.tif') and 'NDVI' in f]
savi_files = [os.path.join(data_dir_SAVI, f) for f in os.listdir(data_dir_SAVI) if f.endswith('.tif') and 'SAVI' in f]

data = []
labels = []

if not ndvi_files or not savi_files:
    print("No NDVI or SAVI files found.")
else:
    min_files = min(len(ndvi_files), len(savi_files))
    ndvi_files = ndvi_files[:min_files]
    savi_files = savi_files[:min_files]

    for ndvi_file, savi_file in zip(ndvi_files, savi_files):
        ndvi = read_raster(ndvi_file)
        savi = read_raster(savi_file)
        # สวัสดีครับอาจารย์ ความหมายของ -9999 หมายถึงในภาพถ่ายดาวเทียมมีเมฆปกคลุมในบริเวณ pixel นั้นๆนะครับผมเลยกรองออกโดยให้เป็น -9999
        nodata_mask = (ndvi != -9999) & (savi != -9999)

        if np.any(nodata_mask):
            combined_data = np.vstack([ndvi[nodata_mask], savi[nodata_mask]]).T
            data.append(combined_data)
            combined_labels = np.random.randint(0, 2, combined_data.shape[0])
            labels.append(combined_labels)
        else:
            print(f"No valid data found in files: {ndvi_file} and {savi_file}")

if data and labels:
    data = np.vstack(data)
    labels = np.hstack(labels)

    X = data
    y = labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = models.Sequential()
    model.add(layers.Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Accuracy: {accuracy*100:.2f}%')
else:
    print("No valid data to train the model.")
