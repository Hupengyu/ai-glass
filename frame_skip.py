"""
视频跳帧处理，可对处理后的视频进行保存
"""

import cv2

# 读取视频
cap = cv2.VideoCapture('./test/1.mp4')

# 跳帧的间隔
skipframe = 10

c = 1

# 视频保存参数
fps_count = cap.get(cv2.CAP_PROP_FPS)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('skipframe.mp4', fourcc, fps_count, (width, height))

while cap.isOpened():
    ret, frame = cap.read()

    # 屏蔽掉下侧else代码后，只显示处理的视频效果
    if c % skipframe == 0:
        # 视频解析处理过程
        # cv2.putText(frame, "HELLO!", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 23, 0), 4, 8)
        cv2.rectangle(frame, (50, 50), (250, 300), (0, 255, 0), 4)
        cv2.imshow('frame', frame)

        # 只保存处理后的视频帧    Q+
        out.write(frame)

    else:
        cv2.imshow('frame', frame)

    # 保存所有视频帧
    out.write(frame)

    c += 1

    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
