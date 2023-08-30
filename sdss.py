# 필요한 모듈 임포트
import os
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import cv2
import time

t1 = time.time()
# 파일 실행 경로로 이동
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# lines.txt의 방출선 목록 읽기(딕셔너리 형태)
with open('lines.txt', 'r') as f:
    element = eval(f.read())

# 흡수선 탐지 알고리즘
def detect_minima(y, sensitivity, mask_pad_size):
    y = np.convolve(y, np.ones(5), 'same') / 5
    y = np.convolve(y, cv2.getGaussianKernel(11, 3).reshape(-1), mode='same')
    diff_mask = [2]
    diff_mask += [0] * mask_pad_size
    diff_mask += [-4]
    diff_mask += [0] * mask_pad_size
    diff_mask += [2]
    conv = np.convolve(y, diff_mask, mode="same")
    points = np.where(conv > sensitivity)[0]
    diff = np.diff(y)
    change_points = []
    for i in points:
        if 1 < i < len(y) - 1 and diff[i-2] < 0 and diff[i - 1] < 0 and diff[i] > 0 and diff[i + 1] > 0:
            change_points.append(i)

    return change_points

# 이동평균 함수 정의
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w

# 스펙트럼 파일 열기(.fits)
data = fits.open('v6_0_4/spec-15143-59205-04544964663.fits')

# 적색편이 보정
x = (10 ** data[1].data['loglam']) / (1 + data[2].data['Z'])

#실제 측정값
ry = data[1].data['FLUX']

# sdss의 후처리가 된 값
y = data[1].data['model']

plt.figure(figsize=(10,5))

# 스펙트럼 그래프
plt.subplot(2, 1, 1)

# x축 범위 지정
plt.xlim(x[0], x[-1])

plt.title('Spectrum Graph')
plt.xlabel('Wavelength(Ångströms)')
plt.ylabel('Flux(10⁻¹⁷ erg/cm²/s/Å)')

# 찾고자 하는 파장 그리기(초록색)
for i in element.keys():
    for j in element[i]:
        plt.axvline(j, color='green', alpha=0.2)

# 원본 데이터 그리기(검은색)
plt.plot(x, ry, 'black', alpha=0.5, label='original flux')

# 보정된 데이터 그리기(빨간색)
ma = moving_average(y, 5)
plt.plot(x, np.convolve(ma, cv2.getGaussianKernel(11, 3).reshape(-1), mode='same'), color='red', label='modified flux')

# 찾은 흡수선 표시(파란색)
find = detect_minima(y, cp.std(y) / 2, 5)
findLine = x[find]
for i in findLine:
    plt.axvline(i, color='blue')

# 흡수선 파장 출력
print(findLine)

# 흡수선 일치 여부 확인 및 원소 출력(오차 ±3A(0.3nm))
for i in element.keys():
    notFound = True
    for line in element[i]:
        notFound = np.any((line - 3 < findLine) & (findLine < line + 3))
        if notFound:
            break
    if notFound:
        print(i)

plt.legend()

#스펙트럼 이미지
plt.subplot(2, 1, 2)

plt.title('Spectrum Image')

# 그래프 이미지화(흑백)
plt.imshow(np.tile(y.reshape(1, -1), (500, 1)), cmap='gray', vmax=np.max(ry), vmin=0)

# 눈금 제거
plt.gca().axes.xaxis.set_visible(False)
plt.gca().axes.yaxis.set_visible(False)

# 이미지에 찾은 흡수선 표시(빨간색)
for i in find:
    plt.axvline(i, color='red', alpha=0.5)

print(time.time() - t1)
plt.show()