import cv2
import numpy as np
import os
import glob

 # folder_name（ここを目的のフォルダに変更してください）
folder_name = "img_0613"

# Defining the dimensions of checkerboard（対象のフォルダ）
CHECKERBOARD = (7,10)
# cv2.TERM_CRITERIA_EPS:指定された精度(epsilon)に到達したら繰り返し計算を終了する
# cv2.TERM_CRITERIA_MAX_ITER:指定された繰り返し回数(max_iter)に到達したら繰り返し計算を終了する
# cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER : 上記のどちらかの条件が満たされた時に繰り返し計算を終了する
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = [] 


# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

# Extracting path of individual image stored in a given directory
script_dir = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(script_dir, folder_name)
if os.path.exists(folder_path):
    print(f"Folder './{folder_name}' exists.")
else:
    print(f"Folder './{folder_name}' does not exist.")
    exit()
images = glob.glob(folder_path + '/*.jpg')
for filepath in images:
    img = cv2.imread(filepath)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+
    	cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    """
    If desired number of corner are detected,
    we refine the pixel coordinates and display 
    them on the images of checker board
    """
    if ret == True:
        objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        
        imgpoints.append(corners2)

        # Draw the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2,ret)
    
    # img_drawChessboardCornersフォルダにチェスボードのコーナー検出画像を保存
    os.makedirs('./img_drawChessboardCorners/', exist_ok=True)
    cv2.imwrite('./img_drawChessboardCorners/' + str(os.path.basename(filepath)), img)
    #cv2.waitKey(0)

# cv2.destroyAllWindows() をコメントアウト
# cv2.destroyAllWindows()

h,w = img.shape[:2]

"""
Performing camera calibration by 
passing the value of known 3D points (objpoints)
and corresponding pixel coordinates of the 
detected corners (imgpoints)
"""
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

print(f"rms: {ret}")
print("Camera matrix : \n")
print(mtx)
print("dist : \n")
print(dist)
# print("rvecs : \n")
# print(rvecs)
# print("tvecs : \n")
# print(tvecs)

# Using the derived camera parameters to undistort the image

for filepath in images:

    img = cv2.imread(filepath)
    # Refining the camera matrix using parameters obtained by calibration
    # ROI:Region Of Interest(対象領域)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    # Method 1 to undistort the image
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # undistort関数と同じ結果が返されるので、今回はコメントアウト(initUndistortRectifyMap()関数)
    # Method 2 to undistort the image
    # mapx,mapy=cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
    # dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)

    # 歪み補正した画像をimg_undistortフォルダへ保存
    os.makedirs('./img_undistort/', exist_ok=True)
    cv2.imwrite('./img_undistort/undistort_' + str(os.path.basename(filepath)), dst)
    cv2.waitKey(0)