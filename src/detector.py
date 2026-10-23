import os
import cv2
import numpy as np
from ultralytics import YOLO

class SafetyDetector:
    def __init__(self, model_name="yolov8n.pt"):
        self.is_custom_ppe = False
        ppe_weights_path = os.path.join("weights", "ppe_yolov8n.pt")
        
        if os.path.exists(ppe_weights_path):
            self.model = YOLO(ppe_weights_path)
            self.is_custom_ppe = True
        else:
            self.model = YOLO(model_name)

        # Background noise classes filter list
        self.ignored_classes = ['bed', 'bird', 'bowl', 'chair', 'dining table', 'couch', 'potted plant']

    def process_frame(self, frame, confidence=0.45):
        results = self.model(frame, conf=confidence)

        detections = []
        annotated_frame = frame.copy()

        h, w, _ = frame.shape
        scale_factor = max(w, h) / 1000.0
        
        line_thickness = max(2, int(2 * scale_factor))
        font_scale = max(0.5, 0.5 * scale_factor)
        font_thickness = max(1, int(1.5 * scale_factor))

        occupied_label_positions = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                class_name = self.model.names[cls_id]

                # Skip background noise objects if using standard model
                if not self.is_custom_ppe and class_name.lower() in self.ignored_classes:
                    continue

                conf_score = float(box.conf[0])
                bbox = box.xyxy[0].tolist()

                x1, y1, x2, y2 = map(int, bbox)

                detections.append({
                    "class": class_name,
                    "confidence": round(conf_score, 2),
                    "bbox": [round(x, 1) for x in bbox]
                })

                # Dynamic Bounding Box Color Scheme
                if class_name.lower() in ['helmet', 'hardhat', 'vest', 'mask', 'safety vest']:
                    box_color = (0, 255, 0) # Green for PPE compliance
                elif class_name.lower() in ['no-helmet', 'no-vest', 'no-mask']:
                    box_color = (0, 0, 255) # Red for hazard violation
                elif class_name == 'person':
                    box_color = (0, 255, 255) # Cyan
                else:
                    box_color = (0, 165, 255) # Orange

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, line_thickness)

                label_text = f"{class_name} {conf_score:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                (text_w, text_h), _ = cv2.getTextSize(label_text, font, font_scale, font_thickness)

                target_y = y1 - int(5 * scale_factor)
                if target_y - text_h < 0 or any(abs(target_y - pos) < text_h + 4 for pos in occupied_label_positions):
                    target_y = y1 + text_h + int(10 * scale_factor)

                occupied_label_positions.append(target_y)

                cv2.rectangle(
                    annotated_frame, 
                    (x1, target_y - text_h - int(4 * scale_factor)), 
                    (x1 + text_w + int(8 * scale_factor), target_y + int(4 * scale_factor)), 
                    box_color, 
                    -1
                )

                cv2.putText(
                    annotated_frame, 
                    label_text, 
                    (x1 + int(4 * scale_factor), target_y), 
                    font, 
                    font_scale, 
                    (0, 0, 0), 
                    font_thickness, 
                    cv2.LINE_AA
                )

        return annotated_frame, detections