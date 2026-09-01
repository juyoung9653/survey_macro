from PyQt6.QtCore import QCoreApplication, QLibraryInfo, QLocale, QTranslator


_STANDARD_BUTTON_TEXT = {
    "OK": "확인",
    "Save": "저장",
    "Save All": "모두 저장",
    "Open": "열기",
    "&Yes": "예",
    "Yes": "예",
    "Yes to &All": "모두 예",
    "Yes to All": "모두 예",
    "&No": "아니요",
    "No": "아니요",
    "N&o to All": "모두 아니요",
    "No to All": "모두 아니요",
    "Abort": "중단",
    "Retry": "다시 시도",
    "Ignore": "무시",
    "Close": "닫기",
    "Cancel": "취소",
    "Discard": "저장 안 함",
    "Help": "도움말",
    "Apply": "적용",
    "Reset": "초기화",
    "Restore Defaults": "기본값 복원",
}


class _KoreanStandardButtonTranslator(QTranslator):
    """Keep standard dialog actions Korean even without Qt's qm bundle."""

    def translate(
        self,
        context: str | None,
        source_text: str | None,
        disambiguation: str | None = None,
        n: int = -1,
    ) -> str | None:
        if context == "QPlatformTheme" and source_text:
            translated = _STANDARD_BUTTON_TEXT.get(source_text)
            if translated is not None:
                return translated
        # None lets Qt continue with the official Korean translator.
        return None


def install_korean_translations(app: QCoreApplication) -> None:
    """Install Korean Qt dialog text once for the current application."""

    if getattr(app, "_korean_translators", None) is not None:
        return

    QLocale.setDefault(QLocale(QLocale.Language.Korean, QLocale.Country.SouthKorea))

    translators: list[QTranslator] = []
    qt_translator = QTranslator(app)
    translations_path = QLibraryInfo.path(
        QLibraryInfo.LibraryPath.TranslationsPath
    )
    if qt_translator.load("qtbase_ko", translations_path):
        app.installTranslator(qt_translator)
        translators.append(qt_translator)

    button_translator = _KoreanStandardButtonTranslator(app)
    app.installTranslator(button_translator)
    translators.append(button_translator)

    # Keep Python references as well as QObject ownership for packaged builds.
    app._korean_translators = translators
