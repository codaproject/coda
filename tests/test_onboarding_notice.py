import logging

from coda.app.onboarding_notice import (
    NOTICE_PLACEHOLDER,
    VERSION_PLACEHOLDER,
    load_onboarding_notice_html,
    render_onboarding_notice,
)


def test_load_onboarding_notice_disabled_ignores_file_path(tmp_path):
    notice_file = tmp_path / "notice.html"
    notice_file.write_text("<section>Demo terms</section>", encoding="utf-8")

    assert load_onboarding_notice_html(False, str(notice_file)) == ""


def test_load_onboarding_notice_reads_enabled_file(tmp_path):
    notice_file = tmp_path / "notice.html"
    notice_file.write_text("<section>Demo terms</section>", encoding="utf-8")

    assert load_onboarding_notice_html(True, str(notice_file)) == (
        "<section>Demo terms</section>"
    )


def test_load_onboarding_notice_missing_file_logs_warning(caplog):
    caplog.set_level(logging.WARNING)

    assert load_onboarding_notice_html(True, "/tmp/does-not-exist.html") == ""
    assert "Could not read onboarding notice file" in caplog.text


def test_render_onboarding_notice_injects_html_and_json_version():
    html = (
        f"<main>{NOTICE_PLACEHOLDER}</main>"
        f"<script>const version = {VERSION_PLACEHOLDER};</script>"
    )

    rendered = render_onboarding_notice(
        html,
        notice_html="<section>Demo terms</section>",
        notice_version='demo-"quoted"',
    )

    assert "<main><section>Demo terms</section></main>" in rendered
    assert 'const version = "demo-\\"quoted\\"";' in rendered
