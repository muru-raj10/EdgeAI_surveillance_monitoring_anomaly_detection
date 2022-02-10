# EdgeAI surveillance monitoring anomaly detection

An Edge AI pipeline to implement learning and inference onto surveillance cameras to detect anomalies such as accidents. 
Objects and motions are extracted from video stream using background subtraction using Gaussian Mixture Model (GMM_BS.py) or (MOG_cv2.py), object detection using YOLO(v3). The anomaly detection model is built on top of it to allow the algorithm/pipeline to run at the Edge or directly at the camera itself.

To explore: online learning of new information at the edge.
