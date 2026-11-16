# 검증 기록

## 확인한 실행

- Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0, PyYAML 6.0.3.
- `PYTHONPATH=src python -m unittest discover -s tests -t tests -q`: 579개 실행, 실패 0, 실제 저장 코퍼스가 필요한 선택적 검사 1개 생략.
- 가상 문서 2개를 6개 원문 청크로 구축하고 검색·후속 질문·전체 요약·6개 검증 사례를 실행.
- 포함한 노트북 4개의 Python 코드를 각각 Run All 두 번에 해당하는 순서로 실행. 네트워크 연결을 차단한 상태로 통과.
- wheel 빌드 후 소스 폴더 밖에서 wheel을 풀어 독립 import와 build/ask/summarize/validate CLI 실행을 확인. JSON 설정·가상 원문·HTML 리소스 포함을 확인.
- 기본 실행의 실제 API 호출 0. 실제 모델의 답변 품질을 측정한 결과가 아님.
- 원문/코드 변경 거절, 문서 범위 제한, 같은 배치 중복 제출 방지, 실제 생성의 키·bool 옵션 검사, 의미 검토 미완료 상태를 확인.

## 별도 확인할 항목

- PyPI가 HTTP 403으로 차단되어 `uv lock`과 새 가상환경 `uv sync`는 완료하지 못함.
- Python 3.11과 실제 Jupyter 커널·브라우저 렌더링은 이번 환경에서 별도 실행하지 않음.
- 실제 OpenAI API, 전체 실제 문서 재색인, 새 문서·질문의 독립 의미 평가, 팀 LangGraph/API 연결은 이번 정리 작업의 실행 검증에 포함하지 않음.

이 패키지는 기존 모델·검색 기준선을 유지한 코드 정리 결과입니다. 새로운 성능 수치를 주장하지 않습니다.
