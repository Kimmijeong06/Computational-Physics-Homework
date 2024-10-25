import os
os.chdir(os.path.abspath(os.path.dirname(__file__)))

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


# hist2.csv 데이터 값 불러오기

energy = []
count = []

with open("hist2.csv", "r") as f:
    for line in f.readlines():
        E, C = [float(i) for i in line.split(",")]
        energy.append(E)
        count.append(C)


# 히스토그램 그리기

plt.hist(energy, bins=energy, weights=count, color='gray')
plt.title("< Energy spectrum of particles A and B >")
plt.xlabel("Energy")
plt.ylabel("Count")
plt.xlim(0,4)
plt.ylim(0,1200)
plt.grid()


# 가우시안 모델 함수

def particle(x, a, mu, sigma):
    y = a*(np.exp(-((x-mu)**2)/(2*sigma**2)))
    return y

def model_function(x, a1, mu1, sigma1, a2, mu2, sigma2):
    particle_A_B = particle(x, a1, mu1, sigma1) + particle(x, a2, mu2, sigma2)
    return particle_A_B


# 초기값 찾기: 최대값(a1, a2), 중심 위치(mu1, mu2)

a = []
mu = []

peaks = find_peaks(count, height=500)[0]

for i in peaks:
    a.append(count[i])
    mu.append(energy[i])

# 초기값 찾기: 표준편차(sigma1, sigma2)

sigma = []

range1 = (mu[0]-0.5, mu[0]+0.5)   # 표준편차를 계산하기 위한 범위(peak위치를 기준으로) 설정
range2 = (mu[1]-0.5, mu[1]+0.5)

energy_range = np.array(energy)

range_value1 = energy_range[(range1[0] <= energy_range) & (energy_range <= range1[1])]
range_value2 = energy_range[(range2[0] <= energy_range) & (energy_range <= range2[1])]

sigma.append(np.std(range_value1))
sigma.append(np.std(range_value2))

# 초기값들(initial values)을 리스트 형태로 저장

initial_values = []

for max_value, mu_value, sigma_value in zip(a, mu, sigma):
    initial_values.append(max_value)
    initial_values.append(mu_value)
    initial_values.append(sigma_value)


# 가우시안 피팅하기

Fitting_para = curve_fit(model_function, energy, count, p0=initial_values)[0]

# 입자 A, B에 대해 가우시안 계산하기

a1_Fit, mu1_Fit, sigma1_Fit = Fitting_para[0], Fitting_para[1], Fitting_para[2]
a2_Fit, mu2_Fit, sigma2_Fit = Fitting_para[3], Fitting_para[4], Fitting_para[5]

FitA = particle(energy, a1_Fit, mu1_Fit, sigma1_Fit)
FitB = particle(energy, a2_Fit, mu2_Fit, sigma2_Fit)


# 입자 A, B에 대해 각각의 가우시안 함수로 피팅된 그래프 그리기

plt.plot(energy, FitA, color='lightgreen', label='Fitted Gaussian for particle A')
plt.plot(energy, FitB, color='skyblue', label='Fitted Gaussian for particle B')
plt.legend()

# 각각의 가우시안 적분

integral_FitA = trapezoid(FitA, energy)
integral_FitB = trapezoid(FitB, energy)


# 입자 A, B에 대해 피팅된 가우시안 함수의 적분값과 생성비(A/B) 출력

print("입자 A의 피팅된 가우시안 함수의 적분값:", integral_FitA)
print("입자 B의 피팅된 가우시안 함수의 적분값:", integral_FitB)
print()
print("생성비(A/B):", integral_FitA/integral_FitB)

plt.show()