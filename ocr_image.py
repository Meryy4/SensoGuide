"""
SensoGuide - OCR Test sur image
Usage: python3 ocr_image.py chemin/vers/image.png
"""

import cv2
import sys
from ocr_module import SensoOCR

if len(sys.argv) < 2:
    print("Usage: python3 ocr_image.py image.png")
    sys.exit(1)

path = sys.argv[1]
frame = cv2.imread(path)

if frame is None:
    print(f"Erreur: impossible de lire '{path}'")
    sys.exit(1)

ocr = SensoOCR()
detections = ocr.read_text(frame)

if not detections:
    print("Aucun texte détecté.")
else:
    print(f"\n{len(detections)} texte(s) détecté(s):\n")
    for d in detections:
        print(f"  → '{d['text']}'  (confiance: {int(d['confidence']*100)}%)")

    all_text = " — ".join([d["text"] for d in detections])
    ocr.speak(all_text, force=True)

    result = ocr.draw_results(frame, detections)
    cv2.imshow("OCR Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
