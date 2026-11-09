# data/

이 폴더의 내용은 GitHub에 올라가지 않습니다 (`.gitignore`). 각자 컴퓨터/서버에 직접 넣어 주세요.

```
data/
├── raw/          # 제공받은 RFP 원본 100건 (hwp, pdf) + 메타데이터 파일
├── processed/    # scripts/build_index.py 가 만드는 파싱 결과 (documents.jsonl)
└── vectorstore/  # scripts/build_index.py 가 만드는 Chroma 벡터DB
```
