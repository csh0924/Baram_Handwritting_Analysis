# Baram Handwriting Analysis

한글 손글씨 분석 AI 프로젝트

## 프로젝트 구조
```
Baram_Handwritting_Analysis/
├── craft/                        # CRAFT 기반 분석
│   ├── common/                  # CRAFT 공통 모듈
│   │   ├── craft.py            # CRAFT 모델
│   │   ├── craft_utils.py      # 유틸리티
│   │   ├── imgproc.py          # 이미지 전처리
│   │   ├── refinenet.py        # Refine 네트워크
│   │   ├── file_utils.py       # 파일 유틸
│   │   └── basenet/            # 베이스 네트워크
│   │
│   └── modules/                # CRAFT 기반 분석 모듈
│       ├── test_original.py   # 원본 (참고용)
│       ├── test_slope.py      # 기울기 분석
│       ├── test_spacing.py    # 자간 분석
│       └── test_size.py       # 크기 분석
│
├── fpn/                         # FPN 기반 분석
│   ├── common/                 # FPN 공통 모듈
│   └── modules/                # FPN 분석 모듈
│       ├── jamo_decomposition.py  # 자음/모음 분해
│       └── jamo_evaluation.py     # 자음/모음 평가
│
└── models/                      # 학습된 모델 가중치 (미포함)
```

## Modules

### CRAFT-based Analysis
- **기울기 분석** (`craft/modules/test_slope.py`)
- **자간 분석** (`craft/modules/test_spacing.py`)
- **크기 분석** (`craft/modules/test_size.py`)

### FPN-based Analysis
- **자음/모음 분해** (`fpn/modules/jamo_decomposition.py`)
- **자음/모음 평가** (`fpn/modules/jamo_evaluation.py`)

## Base Model
- [CRAFT-pytorch](https://github.com/clovaai/CRAFT-pytorch) by Naver Clova

## Installation
```bash
pip install -r requirements.txt
```

## Usage

### 개별 모듈 실행
```bash
# 기울기 분석
python craft/modules/test_slope.py --image sample.jpg

# 자간 분석
python craft/modules/test_spacing.py --image sample.jpg
```

## License
MIT License - See [LICENSE](LICENSE) file

## Acknowledgments
This project is based on CRAFT-pytorch by Naver Clova.
