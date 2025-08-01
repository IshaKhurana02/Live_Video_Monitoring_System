# 📡 Intelligent Live Video Monitoring System

A real-time, scalable, and intelligent CCTV surveillance solution designed to monitor live video streams and detect critical events such as:

- 🚷 Intrusion detection  
- 🌀 Loitering behavior  
- 🧑‍🤝‍🧑 Crowd formation & dispersion  
- 🔥 Fire/smoke detection  
- 😵 Fall detection  
- 🚶 Person in/out tracking  
- 🚫 Wrong direction movement  
- 📢 Directional alarms  

Built for high-performance edge deployment and real-time alerting, this system is engineered to support security, public safety, and automation use cases.

---

## 🚀 Features

- 🎯 Multi-event detection using advanced deep learning models  
- 📦 Real-time object tracking with YOLOv5, Faster R-CNN, and custom detectors  
- ⚡ Low-latency video processing using GStreamer and ZeroMQ (ZMQ)  
- 📡 Multi-stream handling from various IP/CCTV cameras  
- 🧠 Edge-optimized pipelines for resource-efficient deployment  
- 🐳 Dockerized setup for easy deployment across any environment  
- 📚 Alert logging and structured data management via MongoDB  
- 🧩 Modular architecture enabling scalability and plug-n-play components  

---

## 🧠 Tech Stack

| Component           | Technology Used                            |
|---------------------|---------------------------------------------|
| Programming Language| Python                                      |
| Deep Learning       | YOLOv5, Faster R-CNN, Custom CNN models     |
| Video Processing    | OpenCV, GStreamer                           |
| Communication       | ZeroMQ (ZMQ)                                |
| Database            | MongoDB                                     |
| Containerization    | Docker                                      |

---

## 🛠 Architecture

<img width="706" height="344" alt="image" src="https://github.com/user-attachments/assets/9f2f4be3-dfa2-4a51-bba1-23b7ab54f0a7" />

⚙️ Getting Started
Follow these steps to get the system up and running in your local or edge environment.

🔁 1. Clone the Repository

git clone https://github.com/yourusername/intelligent-video-monitoring.git  
cd intelligent-video-monitoring  

🐳 2. Build the Docker Image

docker build -t intelligent-monitoring .

🚀 3. Run the Container

docker run -it --rm -p 5000:5000 intelligent-monitoring  

🛢 4. Configure MongoDB
Ensure MongoDB is installed and running on your machine or hosted remotely.

Update your connection string in the .env or config.json:

MONGODB_URI=mongodb://localhost:27017  
MONGO_DB_NAME=video_monitoring  
💡 Tip: You can use services like MongoDB Atlas for cloud deployment.

## 📊 Real-Time Alerts

All detection events are logged and stored in MongoDB with structured metadata for analysis and visualization. Each alert contains:

- 🕒 **Timestamp** of the event  
- 🎯 **Type of event** (e.g., Intrusion, Fire, Fall)  
- 📷 **Camera ID** or stream source  
- 📍 **Bounding box coordinates** (if applicable)  
- 📡 **Location metadata** (optional, if GPS or zone mapping is enabled)

These records can be:
- Queried for post-event analysis  
- Streamed to dashboards  
- Used for real-time alerts via webhooks or mobile notifications  
- Exported to CSV/JSON for offline reports  

You can easily connect these alerts to a frontend dashboard (Grafana, Streamlit, or custom UI) or integrate with security management systems.

---

## 📈 Current Detection Capabilities

| Event Type         | Detection Status |
|--------------------|------------------|
| 🚷 Intrusion        | ✅ Stable         |
| 🌀 Loitering         | ✅ Stable         |
| 🧑‍🤝‍🧑 Crowd Estimation | ✅ Stable         |
| 🔥 Fire/Smoke       | ✅ Stable         |
| 😵 Fall Detection   | ✅ Stable         |
| 🚶 Person In/Out    | ✅ Stable         |
| 🚫 Wrong Direction  | ✅ Stable         |
| 📢 Directional Alarms | ✅ Stable       |

---

## ⚙️ Performance Highlights

- ⏱ **~30% latency reduction** on edge devices with optimized pipelines  
- 💡 Efficient use of CPU/GPU resources through model quantization and batch inference  
- 🎥 Supports **multi-stream processing** with minimal frame drops  
- 🧠 Edge-ready: runs well on devices like Jetson Nano, Raspberry Pi 4, or Intel NUC  

---

## 🔮 Future Enhancements

Here's a glimpse of what’s planned:

- 📱 **Mobile App**: Real-time alerts and playback on Android/iOS  
- 📊 **Web Dashboard**: Monitor all cameras, events, and analytics in one place  
- 🧠 **Face Recognition Module**: Identify known personnel or blacklisted individuals  
- 🌐 **Cloud Syncing**: Connect multiple locations via a centralized cloud-based hub  
- 🕹️ **Interactive Heatmaps**: Visualize movement, density, and event hotspots  
- 🔄 **Smart Playback & Scheduling**: Replay filtered events on demand  

---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

- 🐞 Report bugs or unexpected behavior  
- 💡 Suggest features or improvements  
- 🔧 Contribute code (detection logic, UI, integrations, etc.)  
- 📚 Help improve documentation

> For large changes, please open an issue first to discuss what you’d like to change.

---

## 👤 Author

**Isha Khurana**  
📧 [ishakhurana@email.com](mailto:ishakhurana@email.com)  
🔗 [linkedin.com/in/isha-khurana](https://linkedin.com/in/isha-khurana)  
💻 [github.com/yourusername](https://github.com/yourusername)

---

## 📄 License

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for full legal terms.

---
