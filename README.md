# 음악정보검색 (Music Information Retrieval) 정리

- 이 레포지토리는 음악정보검색(Music Information Retrieval)을 공부하며 실험한 주피터 노트북(juypter notebook)들로 구성되어 있습니다.
- 대부분의 내용은 책 "Fundamentals of Music Processing: Using Python and Jupyter Notebooks", Müller, Meinard. 의 목차와 내용을 기반으로 합니다. 아래의 [출처](#출처)를 참고해주시길 바랍니다.
- 목차를 클릭하면 해당 노트북으로 링크 이동합니다. Google Colab으로 연결된 노트북을 열람합니다.

## 목차
- [1. 들어가며](https://colab.research.google.com/github/jo-cho/mir_kr/blob/main/Notebooks/1.%20Introduction/01.Introduction.ipynb)

- [2. 음악표현 (Music Representation)](https://github.com/jo-cho/mir_kr/tree/main/Notebooks/2.%20Music%20Representation)
  - [2.1. 악보 (Sheet Music)](https://colab.research.google.com/github/jo-cho/mir_kr/blob/main/Notebooks/2.%20Music%20Representation/2.1.Sheet_Music.ipynb)
  - [2.2. 기호 표현 (Symbolic Representation)](https://colab.research.google.com/github/jo-cho/mir_kr/blob/main/Notebooks/2.%20Music%20Representation/2.2.Symbolic_Representation.ipynb)
  - [2.3. 오디오 표현 (Audio Representation)](https://colab.research.google.com/github/jo-cho/mir_kr/blob/main/Notebooks/2.%20Music%20Representation/2.3.Audio_Representation.ipynb)
  - [2.4. 더 읽을거리](https://colab.research.google.com/github/jo-cho/mir_kr/blob/main/Notebooks/2.%20Music%20Representation/2.4.Further_Readings.ipynb)
  - [2.E. 연습문제](https://colab.research.google.com/github/jo-cho/mir_kr/blob/main/Notebooks/2.%20Music%20Representation/2.E.Exercises.ipynb)
  
- [3. 신호 분석 - 푸리에 분석 (Fourier Analysis of Signals)](https://github.com/jo-cho/mir_kr/tree/main/Notebooks/3.%20Fourier%20Analysis%20of%20Signals)
  - [3.1. 수학 리뷰 - 복소수, 지수함수](https://colab.research.google.com/github/jo-cho/mir_kr/blob/main/Notebooks/3.%20Fourier%20Analysis%20of%20Signals/3.1.Math_Review.ipynb)
  - [3.2. 이산 푸리에 변환 (Discrete Fourier Transform) & FFT](https://colab.research.google.com/github/jo-cho/mir_kr/blob/main/Notebooks/3.%20Fourier%20Analysis%20of%20Signals/3.2.Discrete_Fourier_Transform.ipynb)
  - [3.3. 단기 푸리에 변환 1 (Short-term Fourier Transform, STFT)](https://colab.research.google.com/github/jo-cho/mir_kr/blob/main/Notebooks/3.%20Fourier%20Analysis%20of%20Signals/3.3.Short-term_Fourier_Transform_1.ipynb)
  - [3.4. 단기 푸리에 변환 2](https://colab.research.google.com/github/jo-cho/mir_kr/blob/main/Notebooks/3.%20Fourier%20Analysis%20of%20Signals/3.4.Short-term_Fourier_Transform_2.ipynb)
  - [3.5. 디지털 신호 (Digital Signals)](https://colab.research.google.com/github/jo-cho/mir_kr/blob/main/Notebooks/3.%20Fourier%20Analysis%20of%20Signals/3.5.Digital_Signals.ipynb)
  - [3.6. 더 읽을거리](https://colab.research.google.com/github/jo-cho/mir_kr/blob/main/Notebooks/3.%20Fourier%20Analysis%20of%20Signals/3.6.Further_Readings.ipynb)
  - [3.E. 연습문제](https://colab.research.google.com/github/jo-cho/mir_kr/blob/main/Notebooks/3.%20Fourier%20Analysis%20of%20Signals/3.E.Exercises.ipynb)

- [4. 음악 동기화 (Music Synchronization)](https://github.com/jo-cho/mir_kr/tree/main/Notebooks/4.%20Music%20Synchronization)
  - [4.1. 오디오 동기화 피쳐 (Audio Synchronization Features)](https://colab.research.google.com/github/jo-cho/mir_kr/blob/main/Notebooks/4.%20Music%20Synchronization/4.1.Audio_Synchronization_Features.ipynb)
  - [4.2. 동적 시간 워핑 (Dynamic Time Warping, DTW)](https://colab.research.google.com/github/jo-cho/mir_kr/blob/main/Notebooks/4.%20Music%20Synchronization/4.2.Dynamic_Time_Warping.ipynb)
  - [4.3. 더 읽을거리](https://colab.research.google.com/github/jo-cho/mir_kr/blob/main/Notebooks/4.%20Music%20Synchronization/4.3.Further_Readings.ipynb)
  - [4.E. 연습문제](https://colab.research.google.com/github/jo-cho/mir_kr/blob/main/Notebooks/4.%20Music%20Synchronization/4.E.Exercises.ipynb)

- [5. 음악 구조 분석 (Music Structure Analysis)](https://github.com/jo-cho/mir_kr/tree/main/Notebooks/5.%20Music%20Structure%20Analysis)
  - [5.1. 음악 구조와 분할 (Music_Structure_and_Segmentation)](https://colab.research.google.com/github/jo-cho/mir_kr/blob/main/Notebooks/5.%20Music%20Structure%20Analysis/5.1.Music_Structure_and_Segmentation.ipynb)
  - [5.2. 자기-유사도 행렬 (Self-Similarity Matrix, SSM)](https://colab.research.google.com/github/jo-cho/mir_kr/blob/main/Notebooks/5.%20Music%20Structure%20Analysis/5.2.Self_Similarity_Matrix.ipynb)
  - [5.3. 오디오 썸네일 (Audio Thumbnailing)](https://colab.research.google.com/github/jo-cho/mir_kr/blob/main/Notebooks/5.%20Music%20Structure%20Analysis/5.3.Audio_Thumbnail.ipynb)
  - [5.4. 노벨티 기반 분할 (Novelty-Based Segmentation)](https://colab.research.google.com/github/jo-cho/mir_kr/blob/main/Notebooks/5.%20Music%20Structure%20Analysis/5.4.Novelty-Based_Segmentation.ipynb)
  - [5.5. 평가 방법 (Evaluation)](https://colab.research.google.com/github/jo-cho/mir_kr/blob/main/Notebooks/5.%20Music%20Structure%20Analysis/5.5.Evaluation.ipynb)
  - [5.6. 더 읽을거리](https://colab.research.google.com/github/jo-cho/mir_kr/blob/main/Notebooks/5.%20Music%20Structure%20Analysis/5.6.Further_Readings.ipynb)
  - [5.E. 연습문제](https://colab.research.google.com/github/jo-cho/mir_kr/blob/main/Notebooks/5.%20Music%20Structure%20Analysis/5.E.Exercises.ipynb)

- [6. 화음 인식 (Chord Recognition)](https://github.com/jo-cho/mir_kr/tree/main/Notebooks/6.%20Chord%20Recognition)
  - [6.1. 화성의 기본 이론 (Basic Theory of Harmony)](https://colab.research.google.com/github/jo-cho/mir_kr/blob/main/Notebooks/6.%20Chord%20Recognition/6.1.Basic_Theory_of_Harmony.ipynb)
  - [6.2. 템플릿 기반 화음 인식(Template-Based Chord Recognition)](https://colab.research.google.com/github/jo-cho/mir_kr/blob/main/Notebooks/6.%20Chord%20Recognition/6.2.Template-Based_Chord_Recognition.ipynb)
  - [6.3. HMM 기반 화음 인식(HMM-Based Chord Recognition)](https://colab.research.google.com/github/jo-cho/mir_kr/blob/main/Notebooks/6.%20Chord%20Recognition/6.3.HMM-Based_Chord_Recognition.ipynb)
  - [6.4. 더 읽을거리](https://colab.research.google.com/github/jo-cho/mir_kr/blob/main/Notebooks/6.%20Chord%20Recognition/6.4.Further_Readings.ipynb)
  - [6.E. 연습문제](https://colab.research.google.com/github/jo-cho/mir_kr/blob/main/Notebooks/6.%20Chord%20Recognition/6.E.Exercises.ipynb)

- [7. 템포와 비트 트래킹 (Tempo and Beat Tracking)](https://github.com/jo-cho/mir_kr/tree/main/Notebooks/7.%20Tempo%20and%20Beat%20Tracking)
  - [7.1. 온셋 감지 (Onset Detection)](https://colab.research.google.com/github/jo-cho/mir_kr/blob/main/Notebooks/7.%20Tempo%20and%20Beat%20Tracking/7.1.Onset_Detection.ipynb)
  - [7.2. 템포 분석 (Tempo Analysis)](https://colab.research.google.com/github/jo-cho/mir_kr/blob/main/Notebooks/7.%20Tempo%20and%20Beat%20Tracking/7.2.Tempo_Analysis.ipynb)
  - [7.3. 비트와 펄스 트래킹 (Beat and Pulse Tracking)](https://colab.research.google.com/github/jo-cho/mir_kr/blob/main/Notebooks/7.%20Tempo%20and%20Beat%20Tracking/7.3.Beat_and_Pulse_Tracking.ipynb)
  - [7.4. 더 읽을거리](https://colab.research.google.com/github/jo-cho/mir_kr/blob/main/Notebooks/7.%20Tempo%20and%20Beat%20Tracking/7.4.Further_Readings.ipynb)
  - [7.E. 연습문제](https://colab.research.google.com/github/jo-cho/mir_kr/blob/main/Notebooks/7.%20Tempo%20and%20Beat%20Tracking/7.E.Exercises.ipynb)

- [8. 내용 기반 오디오 검색 (Content-Based Audio Retrieval)](https://github.com/jo-cho/mir_kr/tree/main/Notebooks/8.%20Content-Based%20Audio%20Retrieval)
  - [8.1. 내용 기반 오디오: 서론](https://colab.research.google.com/github/jo-cho/mir_kr/blob/main/Notebooks/8.%20Content-Based%20Audio%20Retrieval/8.1.Introduction.ipynb)
  - [8.2. 오디오 식별 (Audio Identification)](https://colab.research.google.com/github/jo-cho/mir_kr/blob/main/Notebooks/8.%20Content-Based%20Audio%20Retrieval/8.2.Audio_Identification.ipynb)
  - [8.3. 오디오 매칭 (Audio Matching)](https://colab.research.google.com/github/jo-cho/mir_kr/blob/main/Notebooks/8.%20Content-Based%20Audio%20Retrieval/8.3.Audio_Matching.ipynb)
  - [8.4. 버전 식별 (Version Identification)](https://colab.research.google.com/github/jo-cho/mir_kr/blob/main/Notebooks/8.%20Content-Based%20Audio%20Retrieval/8.4.Version_Identification.ipynb)
  - [8.5. 더 읽을거리](https://colab.research.google.com/github/jo-cho/mir_kr/blob/main/Notebooks/8.%20Content-Based%20Audio%20Retrieval/8.5.Further_Readings.ipynb)
  - [8.E. 연습문제](https://colab.research.google.com/github/jo-cho/mir_kr/main/Notebooks/8.%20Content-Based%20Audio%20Retrieval/8.E.Exercises.ipynb)

- [9. 음악 정보에 기반한 오디오 분해 (Musically Informed Audio Decomposition)](https://github.com/jo-cho/mir_kr/tree/main/Notebooks/9.%20Musically%20Informed%20Audio%20Decomposition)
  - [9.1. 화성-타악기 분리 (Harmonic-Percussive Separation)](https://colab.research.google.com/github/jo-cho/mir_kr/blob/main/Notebooks/9.%20Musically%20Informed%20Audio%20Decomposition/9.1.Harmonic%E2%80%93Percussive_Separation.ipynb)
  - [9.2. 멜로디 추출 (Melody Extraction)](https://colab.research.google.com/github/jo-cho/mir_kr/blob/main/Notebooks/9.%20Musically%20Informed%20Audio%20Decomposition/9.2.Melody_Extraction.ipynb)
  - [9.3. NMF 기반 오디오 분해 (NMF-Based Audio Decomposition)](https://colab.research.google.com/github/jo-cho/mir_kr/blob/main/Notebooks/9.%20Musically%20Informed%20Audio%20Decomposition/9.3.NMF-Based_Audio_Decomposition.ipynb)
  - [9.4. 더 읽을거리](https://colab.research.google.com/github/jo-cho/mir_kr/blob/main/Notebooks/9.%20Musically%20Informed%20Audio%20Decomposition/9.4.Further_Readings.ipynb)
  - [9.E. 연습문제](https://colab.research.google.com/github/jo-cho/mir_kr/blob/main/Notebooks/9.%20Musically%20Informed%20Audio%20Decomposition/9.E.Exercises.ipynb)
    
## 노트
* *추가적인 실험을 각 노트북에 추가할 예정이며, 아직 수정중에 있습니다.*
* *오탈자 및 용어 해석의 모호함 등을 고려한 수정이 이루어지고 있습니다.*
* *깃허브 화면에서는 오디오재생이 불가합니다.*
* *오직 교육용 목적입니다.*
* *오디오, 이미지, csv 자료 등은 아래의 출처로부터 얻었습니다.*

## 출처
- Müller, Meinard. Fundamentals of Music Processing: Using Python and Jupyter Notebooks. Springer Nature, 2021.
- https://www.audiolabs-erlangen.de/fau/professor/mueller/bookFMP
- https://musicinformationretrieval.com/
- https://github.com/meinardmueller/libfmp
- https://github.com/SuperShinyEyes/FundamentalsOfMusicProcessing

<img src="https://images-na.ssl-images-amazon.com/images/I/51q5YtafVsL.jpg">

##
편집자: 조정효 <jhcho1016@naver.com>
