# A/B/C/D 비교용 실행 설정

이 파일은 `README_D_PROFILE.md`에 B/C 실행을 추가합니다. 먼저 이전 D 패치를 적용하세요.
이번 추가는 설정·준비 스크립트·검증 파일만 포함합니다. 기존 `src/rfp_rag/`와 A/D 설정은
변경하지 않아, 앞서 구축한 정상 A/D workspace는 그대로 사용할 수 있습니다.

| 조합 | 생성 모델 | 추론 설정 | 검색 | 설정 | 웹 포트 |
| --- | --- | --- | --- | --- | --- |
| A | `gpt-5.4-mini` | `none` | BM25 + 문자 TF-IDF/RRF | `configs/rag_a.yaml` | 8766 |
| B | `gpt-6.1-sol` | `low` | A와 동일 | `configs/rag_b.yaml` | 8767 |
| C | `gpt-5.4-mini` | `none` | 기존 검색 + `text-embedding-3-large`/RRF | `configs/rag_c.yaml` | 8768 |
| D | `gpt-6.1-sol` | `low` | C와 동일 | `configs/rag_d.yaml` | 8765 |

C/D는 문서와 검색질문에 같은 3,072차원 임베딩을 사용합니다.
`configs/rag.yaml`은 앞서 정한 D 설정을 유지합니다. 설정을 생략하는 기존 노트북은 A입니다.

## 1. B/C 준비 (API 호출 없음)

기여 폴더의 PowerShell에서 실행합니다. 원문은 앞서 사용한 것과 같은 JSONL을 지정하세요.

```powershell
Set-Location "C:\Users\yiydg\Desktop\workspace\rfp-agent-part3-team1\contributions\younggwan_rag"
uv sync --extra openai
```

**D의 문서 임베딩이 이미 완료되었다면**, 다음 한 명령으로 B/C 로컬 인덱스를 만들고
D의 검증된 문서 임베딩을 C에 복사합니다. API 키를 읽거나 API를 호출하지 않습니다.

```powershell
uv run python scripts/prepare_bc.py --source workspaces/team-v1/data/source_documents.jsonl --reuse-semantic-from workspaces/team-d-v1
```

기본 출력 경로는 `workspaces/team-b-v1`, `workspaces/team-c-v1`입니다.
원문이 다른 위치에 있으면 `--source data/source_documents.jsonl` 등으로 바꾸세요.
이전에 D를 다른 경로에 구축했다면 `--reuse-semantic-from`도 해당 경로로 지정합니다.

재사용 시 원문 지문·청크 순서·검색 인덱스·임베딩 모델·차원·벡터 파일을 검사합니다.
원본 D와 기존 A는 수정하지 않습니다. 복사한 벡터 파일은 C 소유의 별도 파일이며,
D의 대화 로그·질문 캐시·사용 예산을 가져오지 않습니다. C의 생성 모델은 계속 mini입니다.
임베딩 메타데이터의 사용량은 원래 D에서 구축할 때의 기록이고, 이번 복사의 신규 호출은
출력의 `api_calls_this_run=0`과 `reports/semantic_reuse.json`으로 구분합니다.

같은 준비 명령을 다시 실행하면 B/C와 완료된 임베딩을 검증해서 재사용합니다.
기존 C 임베딩·로그·예산은 덮어쓰지 않습니다. 기존 C 벡터가 D와 다르면 새 C 경로를
요청합니다. C에 미완료 임베딩 시도가 있으면
기록을 덮는 대신 중단하므로 원인을 확인하고 새 `--workspace-c` 경로를 지정하세요.

**D 임베딩이 아직 없다면**, 준비 명령에서 재사용 옵션을 생략합니다.

```powershell
uv run python scripts/prepare_bc.py --source workspaces/team-v1/data/source_documents.jsonl
```

B는 바로 사용할 수 있습니다. C는 `awaiting_embeddings`로 표시되며, 다음 명령으로 문서
임베딩을 구축한 뒤 사용하세요. **아래 `embed --live`는 임베딩 API 비용이 발생합니다.**

```powershell
uv run --env-file .env rfp-rag embed --workspace workspaces/team-c-v1 --live
```

기존 workspace에 원문이나 설정이 다른 경우 준비 스크립트는 덮어쓰지 않습니다.
필요하면 `--workspace-b workspaces/team-b-v2 --workspace-c workspaces/team-c-v2`처럼
새 경로를 명시하세요. B/C 준비 스크립트가 A/D를 새로 구축하지는 않습니다.

## 2. B와 C 웹 실행

각 명령은 **서로 다른 터미널**에서 실행합니다.

B는 로컬 검색을 유지하므로 질문 임베딩 API를 사용하지 않습니다.

```powershell
uv run --env-file .env rfp-rag serve --workspace workspaces/team-b-v1 --port 8767
```

[B 화면](http://127.0.0.1:8767/)에서 문서 선택 → 선택 적용 → **실제 API 답변 생성** 체크 →
**선택 문서 API 설정 적용** 후 질문을 제출합니다. 생성 모델 표시가 `gpt-6.1-sol`인지 확인하세요.

C는 의미 검색을 사용하므로 질문 임베딩을 `--live-search`로 허용합니다.

```powershell
uv run --env-file .env rfp-rag serve --workspace workspaces/team-c-v1 --live-search --port 8768
```

[C 화면](http://127.0.0.1:8768/)에서도 같은 순서로 답변 생성을 켭니다.
생성 모델은 `gpt-5.4-mini`, 의미 검색 모델은 `text-embedding-3-large`로 표시됩니다.
답변 생성을 꺼도 C의 질문 임베딩 비용은 발생할 수 있습니다. 문서 임베딩은 저장본을 씁니다.

기존 A/D 서버는 이전 명령을 그대로 사용합니다. 종료는 각 터미널에서 Ctrl+C입니다.

## 3. 비교할 때 고정할 조건

- **A ↔ B, C ↔ D**: 검색 방식·원문·질문을 고정하고 생성 설정을 비교합니다.
- **A ↔ C, B ↔ D**: 생성 모델·추론 설정을 고정하고 의미 검색 추가 효과를 비교합니다.
- **A ↔ D**: 전체 조합의 변화입니다. 어느 변경이 영향을 줬는지는 위 비교로 확인합니다.

Sol은 `none`을 지원하지 않아 B/D에 `low`를 사용합니다. 따라서 A↔B와 C↔D는
생성 모델과 그 추론 설정을 함께 바꾸는 비교입니다. 순수 모델 차이로만 해석하지 않습니다.
추론 설정까지 맞춘 별도 비교가 필요하면 A/C의 `low` 실험을 새 이름·새 workspace로
추가하고 현재 A/C 기준선은 유지하세요.

같은 문서 ID·필터·질문 순서로 새 대화를 시작하세요. C/D의 문서 임베딩 재사용은
두 조합에 같은 검색 벡터를 제공하지만, 질문 임베딩 호출과 사용 예산은 조합별로 별도입니다.
지연을 비교할 때는 두 서버를 함께 재시작해 메모리의 질문 캐시 조건도 맞추세요.
답변 생성 한도는 세션별로, 질문 임베딩 한도는 workspace 누적으로 관리됩니다.

원문 근거 회수 여부, 핵심 조건 누락, 주장별 인용, 지연과 생성/임베딩 사용량을 확인합니다.
기계 검사 통과나 답변 생성 성공을 의미 정확도 통과로 표시하지 않습니다.

## 4. 로컬 검증

```powershell
uv run python -m unittest discover -s tests -t tests -q
```

추가 검사는 네 프로필의 변경 범위, A/B 및 C/D 요청의 동일 근거·프롬프트,
B의 API 없는 검색, C/D 문서 임베딩 재사용, 손상·불일치 거절, 기존 로그 보존을
모의 클라이언트로 확인합니다. 실제 API 호출이나 실제 RFP의 성능 평가 결과는 아닙니다.
