# Baram Handwriting Analysis

한글 손글씨 분석 AI 프로젝트

## 프로젝트 구조

```text
Baram_Handwriting_Analysis/
├── Analysis_Module/                     # 손글씨 분석 메인 모듈
│   ├── main.py                         # 전체 분석 파이프라인 실행
│   ├── common_paths.py                 # 공통 경로 설정
│   ├── errors.py                       # 사용자 정의 예외
│   │
│   ├── center.py                       # 글자 중심점 검출
│   ├── char_bbox.py                    # 글자 바운딩 박스 추출
│   ├── char_split.py                   # 글자 분리
│   ├── refine_segments.py              # 세그먼트 정제
│   │
│   ├── generate_printed_chars.py       # 기준 문자 생성
│   ├── make_chars_json.py              # 문자 정보 JSON 생성
│   ├── make_json.py                    # 분석 결과 JSON 생성
│   ├── make_score.py                   # 종합 점수 생성
│   │
│   ├── normalize_1.py                  # 1차 정규화
│   ├── normalize_2.py                  # 2차 정규화
│   │
│   ├── score_size.py                   # 크기 점수
│   ├── score_space.py                  # 자간 점수
│   ├── score_tilt.py                   # 기울기 점수
│   ├── score_jamo.py                   # 자모 점수
│   │
│   ├── space.py                        # 자간 분석
│   ├── tilt.py                         # 기울기 분석
│   ├── io_unicode_cv.py                # Unicode/OpenCV 입출력
│   ├── jamo_eval.py                    # 자모 평가
│   ├── fpn_segment.py                  # FPN 기반 세그멘테이션
│   │
│   └── craft_module/
│
├── Develop/                            # 연구·실험용 코드
│   │
│   ├── craft/
│   │   ├── common/
│   │   │   ├── basenet/
│   │   │   │   ├── __init__.py
│   │   │   │   └── vgg16_bn.py
│   │   │   ├── craft.py
│   │   │   ├── craft_utils.py
│   │   │   ├── file_utils.py
│   │   │   ├── imgproc.py
│   │   │   └── refinenet.py
│   │   │
│   │   └── modules/
│   │       ├── CRAFT_center.ipynb
│   │       ├── Visualize_1.ipynb
│   │       ├── Visualize_3.ipynb
│   │       ├── char_bbox_module.ipynb
│   │       ├── char_split_module.ipynb
│   │       ├── normalize_module.ipynb
│   │       ├── normalize_module_2.ipynb
│   │       ├── space_module.ipynb
│   │       └── tilt_module.ipynb
│   │
│   ├── fpn/
│   │   └── modules/
│   │       ├── cluster.ipynb
│   │       ├── fpn_common.ipynb
│   │       ├── fpn_infer.ipynb
│   │       ├── fpn_train.ipynb
│   │       ├── generate_sentence_chars.ipynb
│   │       ├── jamo_module.ipynb
│   │       └── suppress.ipynb
│   │
│   └── hangul_dataset/
│       ├── generate_base_char.ipynb
│       ├── generate_custom_char.ipynb
│       ├── generate_custom_char_temp.ipynb
│       ├── korean_char_frequency_analysis.ipynb
│       └── labeling_chojungjong.ipynb
│
├── server.py                           # Flask API 서버
├── README.md
├── LICENSE
└── .gitignore
```

## 주요 기능

### 문자 검출 및 분리

* CRAFT 기반 텍스트 영역 검출
* 글자 중심점 추출
* 글자 단위 분할
* 문자 바운딩 박스 생성

### 손글씨 분석

* 글자 크기 분석
* 글자 기울기 분석
* 자간 분석
* 자모 단위 분석

### 평가 및 점수화

* 크기 점수 산출
* 기울기 점수 산출
* 자간 점수 산출
* 자모 점수 산출
* 종합 점수 계산

### 결과 생성

* 분석 결과 JSON 생성
* 문자별 메타데이터 생성
* API 응답용 데이터 생성

## 실행 구조

```text
Client
  ↓
server.py
  ↓
Analysis_Module.main
  ↓
문자 검출
  ↓
글자 분리
  ↓
정규화
  ↓
크기/기울기/자간/자모 분석
  ↓
점수 계산
  ↓
analysed.json
chars.json
```

## API 서버 실행

```bash
python server.py
```

기본 실행 주소:

```text
http://localhost:5000
```

### Health Check

```http
GET /health
```

응답:

```json
{
  "ok": true
}
```

### 손글씨 분석

```http
POST /analyze
```

Form Data

| Field          | Description |
| -------------- | ----------- |
| handwriting_id | 분석 ID       |
| original_text  | 원문          |
| font           | 기준 폰트       |
| image          | 손글씨 이미지     |

응답:

```json
{
  "analysed": {...},
  "chars": {...}
}
```

## 개발 디렉토리

`Develop/` 디렉토리는 실제 서비스 코드가 아닌 연구 및 실험용 Notebook을 보관합니다.

* CRAFT 실험
* FPN 실험
* 한글 데이터셋 생성
* 시각화 및 검증

## Base Models

* CRAFT (Character Region Awareness for Text Detection)
* FPN (Feature Pyramid Network)

## Installation

```bash
pip install -r requirements.txt
```

## License

MIT License

See [LICENSE](LICENSE) for details.
