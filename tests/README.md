# 테스트 실행

일반 테스트는 외부 PDF가 없어도 실행됩니다. `OK (skipped=...)`는 실물 PDF 검증까지 통과했다는 뜻이 아닙니다.

```powershell
python -X utf8 -m unittest discover -s tests -v
```

리더·마음·여행을 함께 불러오는 실물 회귀 테스트:

```powershell
$env:RUN_TRIO_CORPUS_TESTS = '1'
python -X utf8 -m unittest tests.test_scan_corpus.IndependentFileLayoutCorpusTests -v
```

이 테스트는 실제 UI에서 세 파일을 선택하고 `기본` 프리셋을 적용한 뒤 분석합니다. 검토용 PDF 83쪽, 파일별 첫째·둘째·마지막 페이지의 답안 칸 위치, 빈 리더 설문 2쪽을 제외한 엑셀 81건을 검사합니다. 모든 필기 답안의 정답 여부까지 검증하는 테스트는 아닙니다.

기존 파일의 개별·2개·3개·전체 묶음과 위 세 파일 회귀 테스트를 모두 실행하려면:

```powershell
$env:RUN_SCAN_CORPUS_TESTS = '1'
python -X utf8 -m unittest tests.test_scan_corpus -v
```

기본 자료 위치는 `C:\Users\Public\scan`, 프리셋 위치는 `%LOCALAPPDATA%\CheckFinder\presets`입니다. 각각 `SURVEY_SCAN_CORPUS`, `SURVEY_PRESET_DIR` 환경 변수로 바꿀 수 있습니다. 전체 테스트는 PDF 목록이 바뀌면 실패하므로 새 파일을 검사 대상으로 분류해야 합니다. 접근 거부나 자료 누락으로 실행하지 못한 검사는 통과로 보고하지 않습니다.
