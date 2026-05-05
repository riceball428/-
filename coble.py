import cv2
import numpy as np
import math
# 【關鍵修改 1】載入 Colab 專用的 imshow 模組
from google.colab.patches import cv2_imshow 

# 載入 OpenCV 內建的 Haar 特徵分類器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def process_eye_and_find_pupil(eye_img):
    """
    對眼睛區域進行影像處理以找出瞳孔中心與半徑
    """
    # 1. 高斯模糊 (Gaussian Blur)
    blur = cv2.GaussianBlur(eye_img, (7, 7), 0)
    
    # 2. 二值化 (Binarization): 擷取暗部瞳孔
    _, thresh = cv2.threshold(blur, 40, 255, cv2.THRESH_BINARY_INV)

    # 3. 尋找輪廓 (Contour)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # 找出面積最大的暗塊 (假設為瞳孔)
        max_contour = max(contours, key=cv2.contourArea)
        # 取得外接圓
        (x, y), radius = cv2.minEnclosingCircle(max_contour)
        return (int(x), int(y)), int(radius)
    
    return None, None

def main():
    # 讀取影像 (請確保圖片已經上傳到 Colab 左側檔案區)
    img_path = 'sample_data/test_face.jpg' 
    img = cv2.imread(img_path) 
    
    if img is None:
        print(f"⚠️ 找不到圖片 '{img_path}'！請檢查是否已將圖片拖曳上傳至左側資料夾。")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 偵測人臉
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    pupil_centers = []

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        # 偵測眼睛
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        # 取前兩個偵測到的眼睛
        for (ex, ey, ew, eh) in eyes[:2]:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            eye_roi_gray = roi_gray[ey:ey+eh, ex:ex+ew]
            
            # 找瞳孔
            center, radius = process_eye_and_find_pupil(eye_roi_gray)
            
            if center is not None:
                # 換算回整張圖的座標
                global_center = (x + ex + center[0], y + ey + center[1])
                pupil_centers.append(global_center)
                
                # 畫出瞳孔範圍
                cv2.circle(img, global_center, radius, (0, 0, 255), 2)
                cv2.circle(img, global_center, 2, (0, 255, 255), 3)

    # 計算瞳孔距離
    if len(pupil_centers) >= 2:
        pt1 = pupil_centers[0]
        pt2 = pupil_centers[1]
        distance = math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
        
        cv2.line(img, pt1, pt2, (255, 255, 0), 2)
        cv2.putText(img, f"Pupil Dist: {distance:.2f} px", (pt1[0], pt1[1] - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        print(f"✅ 兩眼瞳孔中心距離: {distance:.2f} 像素")

    # 【關鍵修改 2】使用 Colab 專用的 cv2_imshow 顯示圖片
    # 取代了原本的 cv2.imshow, cv2.waitKey(0), cv2.destroyAllWindows()
    cv2_imshow(img)

if __name__ == '__main__':
    main()