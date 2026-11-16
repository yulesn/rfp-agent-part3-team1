# D 조합 실행과 A 기준선 보존

D는 `gpt-6.1-sol`의 근거 답변과 OpenAI 의미 검색을 연결합니다. A는 이전 모델과 검색을 유지합니다. 두 설정을 같은 원문으로 구축하되 workspace와 실행 기록을 분리합니다.

| 구분 | A 기준선 | D 조합 |
| --- | --- | --- |
| 설정 | `configs/rag_a.yaml` | `configs/rag_d.yaml` 또는 `configs/rag.yaml` |
| 답변 모델 | `gpt-5.4-mini` | `gpt-6.1-sol` |
| 추론 설정 | `none` | `low` |
| 검색 | BM25 + 문자 TF-IDF/RRF | BM25 + 문자 TF-IDF + 의미 검색/RRF |
| 의미 검색 모델 | 사용하지 않음 | `text-embedding-3-large`, 3,072차원 |
| 권장 workspace | `workspaces/team-a-v1` | `workspaces/team-d-v1` |

`gpt-6.1-sol`은 추론 설정 `none`을 지원하지 않아 `low`를 명시합니다. 질문과 문서는 같은 임베딩 모델·차원으로 변환합니다. BERT/BGE 모델은 이번 변경에 포함하지 않았습니다.

## 1. 같은 원문으로 A와 D 구축

PowerShell에서 기여 폴더로 이동합니다. `.env`의 `OPENAI_API_KEY`는 기존 개인 설정을 사용합니다.

```powershell
Set-Location "C:\Users\yiydg\Desktop\workspace\rfp-agent-part3-team1\contributions\younggwan_rag"
uv sync --extra openai

uv run rfp-rag build --source data/source_documents.jsonl --workspace workspaces/team-a-v1 --config configs/rag_a.yaml
uv run rfp-rag build --source data/source_documents.jsonl --workspace workspaces/team-d-v1 --config configs/rag_d.yaml
```

위 `build` 두 명령은 청킹과 로컬 검색 인덱스만 구축하며 API를 호출하지 않습니다. D의 문서 임베딩은 다음 단계에서 별도로 생성합니다.

원문이 기존 workspace에만 있다면 두 명령의 `--source`를 모두 `workspaces/team-v1/data/source_documents.jsonl`로 바꾸세요. 입력은 PDF 자체가 아닌 파싱된 원문 JSONL입니다.

기존 `workspaces/team-v1`은 삭제하거나 덮어쓰지 않습니다. 구현 코드가 바뀌면 기존 workspace의 코드 지문 검사가 실패하므로 새 A와 D 경로를 사용하세요. 이전 결과와 로그는 기존 경로에 보존됩니다. 설정을 생략하는 기존 Python/노트북 실행은 계속 A를 사용합니다. D는 설정 파일을 명시해서 구축하세요.

## 2. D 문서 임베딩 생성

```powershell
uv run --env-file .env rfp-rag embed --workspace workspaces/team-d-v1 --live
```

이 단계는 **D workspace에 포함된 문서 청크를 OpenAI 임베딩 API로 전송**하며 API 비용이 발생합니다. 답변 생성 모델은 호출하지 않습니다. 같은 원문·모델·차원의 완료 결과는 재사용합니다. 처리 중 오류나 결과가 불확실한 요청은 자동 재시도하지 않습니다. 오류가 나면 기록을 확인하고, 재개 목적으로 기존 기록을 임의로 삭제하지 마세요.

모델이나 차원을 바꾸려면 설정을 변경하고 새 workspace를 구축하세요. A에는 이 단계를 실행할 필요가 없습니다.

## 3. 웹에서 D 실행

```powershell
uv run --env-file .env rfp-rag serve --workspace workspaces/team-d-v1 --live-search --port 8765
```

[http://127.0.0.1:8765/](http://127.0.0.1:8765/)를 열고 문서 선택 → 선택 적용 → 질문 제출 순서로 사용합니다. 답변까지 받으려면 왼쪽 **답변 생성 설정**에서 **실제 API 답변·요약 생성**을 체크하고 **선택 문서 API 설정 적용**을 누르세요. 문서 하나를 선택하면 오른쪽에서 예상 호출 수를 확인하고 **전체 요약 생성**도 실행할 수 있습니다. 요약은 질문 한도와 별도로 기록되며, 이 웹 요약 업데이트를 위해 문서 임베딩을 다시 만들 필요는 없습니다.

`--live-search`는 질문 임베딩 API를 허용하는 옵션입니다. 웹의 답변 생성 체크와 별개이므로 **답변 생성을 꺼도 검색 질문의 임베딩 비용은 발생할 수 있습니다.** 문서 임베딩은 저장된 결과를 사용합니다. D에서 API 사용을 허용하지 않으면 의미 검색을 조용히 생략하지 않고 실행을 거절합니다.

다른 터미널에서 A를 별도 포트로 실행하면 두 결과를 나란히 확인할 수 있습니다.

```powershell
uv run --env-file .env rfp-rag serve --workspace workspaces/team-a-v1 --port 8766
```

A 화면은 [http://127.0.0.1:8766/](http://127.0.0.1:8766/)입니다. A의 로컬 검색에는 API 비용이 없고, 답변 생성을 켜면 `gpt-5.4-mini`를 호출합니다. 서버 종료는 각 터미널에서 Ctrl+C입니다.

## 4. CLI에서 D 확인

먼저 문서 ID를 확인합니다. `catalog`는 API를 호출하지 않습니다.

```powershell
uv run rfp-rag catalog --workspace workspaces/team-d-v1
```

아래 문서 ID는 실제 대상에 맞게 바꾸세요. `search --live`는 질문 임베딩만 실행하고, `ask --live`는 질문 임베딩과 답변 생성을 함께 허용합니다.

```powershell
uv run --env-file .env rfp-rag search --workspace workspaces/team-d-v1 --doc-id "1aYJqERWbnhaMkOlJTrT5BDxydDWTRmUs" --question "완료·검수 조건은?" --live
uv run --env-file .env rfp-rag ask --workspace workspaces/team-d-v1 --doc-id "1aYJqERWbnhaMkOlJTrT5BDxydDWTRmUs" --question "완료·검수 조건은?" --live
```

## 5. 검증과 기록 해석

```powershell
uv run python -m unittest discover -s tests -t tests -q
```

자동 검사는 가상 문서와 모의 API 응답으로 연결·문서 범위·인용 보존·오류 처리를 확인합니다. 실제 OpenAI API의 성공이나 실제 RFP의 의미 정확도를 검증한 결과는 아닙니다.

문서 임베딩 작업 기록은 workspace의 `semantic_build/`, 질문 임베딩 기록은 `semantic_outputs/`에 저장합니다. 임베딩 호출 수와 사용량은 기존 LLM 답변 생성 호출 수와 별개입니다. D 설정의 질문 임베딩 상한은 workspace 누적 64회·256,000 입력 바이트이며, 한 질문을 여러 검색질문으로 나누면 여러 회를 사용할 수 있습니다. 같은 프로세스의 동일 검색질문은 캐시를 재사용합니다. 서버를 재시작해도 누적 예산 기록은 유지됩니다. 기존 `phase*_outputs/` 답변 기록과 함께 확인하세요.

A와 D를 같은 문서·질문으로 비교하면 전체 조합의 차이를 볼 수 있습니다. 생성 모델, 추론 설정, 검색이 함께 달라지므로 개선 원인을 구분하려면 이후 B/C 실험도 필요합니다. 정답 근거 검색 여부, 답변의 누락·왜곡, 주장별 인용, 지연과 사용량을 대조하세요. 기계 검사 통과를 의미 정확도 통과로 표시하지 않습니다.

공식 문서: [GPT-6.1 Sol](https://developers.openai.com/api/docs/models/gpt-6.1-sol), [OpenAI 임베딩](https://developers.openai.com/api/docs/guides/embeddings).
