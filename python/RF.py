import numpy as np
import rasterio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import os

def read_raster(file_path):
    with rasterio.open(file_path) as src:
        return src.read(1)

data_dir_SAVI = '../data/Index/savi'
data_dir_NDVI = '../data/Index/ndvi'

ndvi_files = [os.path.join(data_dir_NDVI, f) for f in os.listdir(data_dir_NDVI) if f.endswith('.tif') and 'NDVI' in f]
savi_files = [os.path.join(data_dir_SAVI, f) for f in os.listdir(data_dir_SAVI) if f.endswith('.tif') and 'SAVI' in f]

ndvi = read_raster(ndvi_files[0])
savi = read_raster(savi_files[0])
# สวัสดีครับอาจารย์ ความหมายของ -9999 หมายถึงในภาพถ่ายดาวเทียมมีเมฆปกคลุมในบริเวณ pixel นั้นๆนะครับผมเลยกรองออกโดยให้เป็น -9999
nodata_mask = (ndvi != -9999) & (savi != -9999)
data = np.vstack([ndvi[nodata_mask], savi[nodata_mask]]).T

labels = np.random.randint(0, 2, data.shape[0])

X = data
y = labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=30)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy*100:.2f}%')
print(classification_report(y_test, y_pred))