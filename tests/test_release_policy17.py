"""Packaging compatibility only; fixed resources use neutral bytes, not cases."""
import hashlib
import json
import zipfile

import pytest
from scripts import build_minimal_release as release


def fixture_source(tmp_path):
    source = tmp_path / 'source'
    for name in release._required_files(release.POLICY_VERSION):
        p = source / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('{}\n' if name.endswith('.json') else '# neutral package fixture\n', encoding='utf-8')
    return source


def test_asr_document_is_required_only_for_current_source_package(tmp_path):
    source = fixture_source(tmp_path)
    (source / 'docs/ASR_LOCAL.md').unlink()
    with pytest.raises(release.ReleaseError, match='required_source_missing:docs/ASR_LOCAL.md'):
        release.build_release(source, tmp_path / 'missing.zip')
    assert 'docs/ASR_LOCAL.md' not in release._required_files(release.V16_POLICY_VERSION)


def test_existing_asr_document_roundtrips_and_private_audio_does_not(tmp_path):
    source = fixture_source(tmp_path)
    output = tmp_path / 'current.zip'
    release.build_release(source, output)
    assert release.verify_release(output)['policy_version'] == 'aria-minimal-source-17'
    with zipfile.ZipFile(output) as archive:
        assert archive.read('docs/ASR_LOCAL.md') == (source / 'docs/ASR_LOCAL.md').read_bytes()
    assert not release._allowed('docs/private.wav')
    assert not release._allowed('artifacts/local/asr-result.json')
    assert not release._allowed('docs/ASR_LOCAL.md', release.V16_POLICY_VERSION)


def test_v16_archive_retains_its_original_scope_and_four_resources(tmp_path):
    source = fixture_source(tmp_path)
    current = tmp_path / 'current.zip'
    release.build_release(source, current)
    with zipfile.ZipFile(current) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    del entries['docs/ASR_LOCAL.md']
    manifest = json.loads(entries['RELEASE_MANIFEST.json'])
    manifest['policy_version'] = release.V16_POLICY_VERSION
    manifest['files'] = [row for row in manifest['files'] if row['path'] != 'docs/ASR_LOCAL.md']
    entries['RELEASE_MANIFEST.json'] = release._manifest_bytes(manifest)
    entries['RELEASE_SCOPE.md'] = release.V16_SCOPE_TEXT.encode('utf-8')
    old = tmp_path / 'old16.zip'
    with zipfile.ZipFile(old, 'w') as archive:
        for name, data in entries.items(): archive.writestr(name, data)
    assert release.verify_release(old)['policy_version'] == release.V16_POLICY_VERSION
    assert all(name in entries for name in release.V16_POLICY_FILES)
    original = hashlib.sha256(old.read_bytes()).hexdigest()
    release.verify_release(old)
    assert hashlib.sha256(old.read_bytes()).hexdigest() == original
