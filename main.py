import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

# 載入 Iris 數據集
data = load_iris()
X = data.data
y = to_categorical(data.target)  # 將標籤轉換為 one-hot 編碼
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 訓練集與測試集分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 構建模型
model = Sequential([
    Dense(64, activation='relu', input_dim=4),  # 輸入層與隱藏層
    BatchNormalization(),  # 批量正規化
    Dropout(0.2),  # Dropout 層
    Dense(32, activation='relu'),  # 隱藏層
    BatchNormalization(),  # 批量正規化
    Dropout(0.2),  # Dropout 層
    Dense(3, activation='softmax')  # 輸出層 (3 類)
])

# 編譯模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 回調函數: EarlyStopping 和 Learning Rate Scheduler
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1)

# 訓練模型
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32, 
                    callbacks=[early_stopping, lr_scheduler])

# 評估模型
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"測試集準確率: {test_acc:.4f}")
