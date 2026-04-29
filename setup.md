⚙️ Setup Instructions (Windows)
1. Clone the repository
git clone <YOUR_REPO_URL>
cd <REPO_NAME>
2. Create virtual environment
python -m venv tf_env
tf_env\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
▶️ Run the System
python pipelines/live_moe.py
🎯 Expected Output
Webcam will open
System will detect appliances in real-time
Bounding boxes will appear
Labels will switch between:
kitchen
display
climate
🛠️ Troubleshooting
❌ Camera not opening

Change in code:

cv2.VideoCapture(0)

to:

cv2.VideoCapture(1)
❌ No detections
Ensure good lighting
Try showing clear objects (TV, fridge, AC)
Lower threshold in code (0.4 → 0.3)
❌ Slow performance
This runs on CPU
Some lag is normal
📌 Notes
Press ESC to exit the application
Make sure webcam access is enabled
Use Python 3.10 for best compatibility
🧠 Architecture (Simple)
Camera → Router → Expert Model → Detection Output
🚀 Future Improvements
Add Utility Expert (washing machine, robot vacuum)
Replace rule-based router with scene classifier
Deploy to Android (MediaPipe / TFLite)