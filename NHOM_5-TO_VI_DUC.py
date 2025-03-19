import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

class HousePriceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dự đoán Giá Nhà")
        self.root.geometry("600x700")
        self.root.configure(bg="#e6f7ff")
        
        self.model = None
        self.scaler = None
        
        ttk.Label(root, text="🏠 Dự đoán Giá Nhà 🏠", font=("Arial", 16, "bold"), background="#e6f7ff", foreground="#003366").pack(pady=10)
        
        self.btn_load = ttk.Button(root, text="📂 Chọn file CSV", command=self.load_data)
        self.btn_load.pack(pady=10)
        
        self.btn_train = ttk.Button(root, text="📊 Huấn luyện mô hình", command=self.train_model, state=tk.DISABLED)
        self.btn_train.pack(pady=10)
        
        self.frame_inputs = ttk.LabelFrame(root, text="📌 Thông tin nhà", padding=10)
        self.frame_inputs.pack(pady=10, padx=10, fill="both")
        
        self.entry_vars = {}
        fields = ["Năm bán", "Tuổi nhà", "Khoảng cách đến TP", "Số cửa hàng", "Vĩ độ", "Kinh độ"]
        
        for field in fields:
            frame = ttk.Frame(self.frame_inputs)
            frame.pack(fill="x", pady=5)
            label = ttk.Label(frame, text=field, width=20)
            label.pack(side=tk.LEFT)
            entry = ttk.Entry(frame, width=20)
            entry.pack(side=tk.RIGHT, expand=True, fill="x")
            self.entry_vars[field] = entry
        
        self.btn_predict = ttk.Button(root, text="📈 Dự đoán giá nhà", command=self.predict_price, state=tk.DISABLED)
        self.btn_predict.pack(pady=10)
        
        self.btn_plot = ttk.Button(root, text="📊 Hiển thị biểu đồ", command=self.show_plots, state=tk.DISABLED)
        self.btn_plot.pack(pady=10)
        
        self.result_label = ttk.Label(root, text="", font=("Arial", 12, "bold"), background="#e6f7ff", foreground="blue")
        self.result_label.pack(pady=10)
    
    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return
        
        self.df = pd.read_csv(file_path)
        self.df.columns = ["ID", "Sale_Year", "House_Age", "Distance_to_City", "Num_Stores", "Latitude", "Longitude", "Price"]
        self.df.drop(columns=["ID"], inplace=True)
        
        self.btn_train.config(state=tk.NORMAL)
        self.btn_plot.config(state=tk.NORMAL)
        messagebox.showinfo("Thông báo", "Dữ liệu đã được tải thành công!")
    
    def train_model(self):
        X = self.df[['Sale_Year', 'House_Age', 'Distance_to_City', 'Num_Stores', 'Latitude', 'Longitude']]
        y = self.df["Price"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model = Sequential([
            Input(shape=(X_train.shape[1],)),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.history = self.model.fit(X_train_scaled, y_train, epochs=50, batch_size=16, validation_data=(X_test_scaled, y_test))
        
        self.model.save("house_price_model.keras")
        self.btn_predict.config(state=tk.NORMAL)
        messagebox.showinfo("Thông báo", "Huấn luyện mô hình thành công!")
    
    def predict_price(self):
        try:
            input_values = [float(self.entry_vars[field].get()) for field in self.entry_vars]
            new_house = np.array([input_values])
            new_house_scaled = self.scaler.transform(new_house)
            predicted_price = self.model.predict(new_house_scaled)[0][0]
            self.result_label.config(text=f"Giá dự đoán: {predicted_price:,.2f} VND")
        except Exception as e:
            messagebox.showerror("Lỗi", "Vui lòng nhập đúng dữ liệu!")
    
    def show_plots(self):
        plt.figure(figsize=(10, 5))
        sns.histplot(self.df["Price"], bins=50, kde=True)
        plt.xlabel("Giá nhà")
        plt.ylabel("Tần suất")
        plt.title("Biểu đồ phân phối giá nhà")
        plt.show()
        
        plt.figure(figsize=(8, 5))
        sns.scatterplot(x=self.df["House_Age"], y=self.df["Price"])
        plt.xlabel("Tuổi nhà (năm)")
        plt.ylabel("Giá nhà")
        plt.title("Mối quan hệ giữa tuổi nhà và giá nhà")
        plt.show()
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(self.df.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
        plt.title("Ma trận tương quan giữa các biến")
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = HousePriceApp(root)
    root.mainloop()
