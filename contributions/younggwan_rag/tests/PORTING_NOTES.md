# 회귀 검사 이동

기존 가상 원문 회귀 검사를 새 패키지 import 경로로 이동했습니다.
과거 노트북 파일명·패치 스크립트 구조 검사는 제외하고 새 공개 API/CLI와 노트북 검사를 추가합니다.

- test_pilot.py:test_http_schema_error_is_visible_in_cli_and_saved_session_report (old removed CLI; covered by new CLI tests)
- test_pilot.py:test_diagnostics_cli_reads_saved_records_without_bootstrap_key_or_model (old removed CLI; covered by new CLI tests)
- test_validation.py:test_cli_diagnostics_only_reads_existing_session_with_api_disabled (old removed CLI; covered by new CLI tests)
