import cv2

cv2.imwrite(
    "/home/kyle/repos/Autonomous-Navigation/readme_imgs/hardware.jpeg",
    cv2.cvtColor(
        cv2.imread("/home/kyle/repos/Autonomous-Navigation/readme_imgs/hardware.jpeg"),
        cv2.COLOR_BGR2RGB,
    ),
)
